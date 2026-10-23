"""Two-replica role-loss throughput calibration; not an alternating training run.

Uses a fixed standard-data probe to isolate context-dependent model-update cost.
KL and probe-refit phases are deliberately excluded and must be timed separately.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from role_probe_training.probes.harmony import ROLES
from role_probe_training.probes.linear import RoleProbe
from role_probe_training.trajectories.replay import encode_record


def encode_examples(path, tokenizer, cap):
    examples = []
    for line in path.open():
        row = json.loads(line)
        encoded = encode_record(tokenizer, row)
        labels = np.asarray(encoded['labels']).copy()
        for i, mid in enumerate(encoded['message_ids']):
            if mid < 0 or encoded['source_message_indices'][int(mid)] < 0:
                labels[i] = -1
        examples.append(dict(id=row.get('trajectory_id', row.get('sample_id')),
                             ids=list(encoded['input_ids']), labels=labels.tolist()))
    # Identical seeded batches in every context condition. Longest is tested separately.
    longest = max(examples, key=lambda x: len(x['ids']))
    random.Random(123).shuffle(examples)
    for example in examples:
        if cap:
            example['ids'] = example['ids'][:cap]
            example['labels'] = example['labels'][:cap]
    return examples, longest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='.cache/models/gpt-oss-20b')
    p.add_argument('--probe', default='experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz')
    p.add_argument('--context', type=int, required=True, help='0 means full trajectory')
    p.add_argument('--updates', type=int, default=3, help='First update is warmup')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--microbatch-size', type=int, default=1)
    p.add_argument('--length-grouped', action='store_true')
    p.add_argument('--check-padding', action='store_true')
    args = p.parse_args()
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    if args.microbatch_size < 1 or args.batch_size % (world * args.microbatch_size):
        raise ValueError('Effective batch must be divisible by replicas times microbatch size')
    torch.manual_seed(123)
    args.output.mkdir(parents=True, exist_ok=True)
    def emit(record):
        print(json.dumps(record), flush=True)
        with (args.output / f'rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
    emit({'event': 'starting', 'rank': rank, 'context': args.context,
          'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    examples, longest = encode_examples(args.data, tokenizer, args.context)
    probe = RoleProbe.load(args.probe)
    assert probe.metadata['roles'] == list(ROLES)
    assert probe.metadata['layer'] == 12
    assert probe.metadata['activation'] == 'post_attention_layernorm_output_pre_mlp'
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': rank}, quantization_config=Mxfp4Config(dequantize=True),
        attn_implementation='flex_attention')
    model = get_peft_model(base, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0., bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=1e-4)
    weight = torch.tensor(probe.weight, device=rank, dtype=torch.float32)
    bias = torch.tensor(probe.bias, device=rank, dtype=torch.float32)
    captured = {}
    def capture(module, inputs, output):
        captured['hidden'] = output
    handle = base.model.layers[12].post_attention_layernorm.register_forward_hook(capture)
    emit({'event': 'loaded', 'seconds': time.perf_counter()-started,
          'trainable_parameters': sum(p.numel() for p in parameters),
          'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
          'scope': 'role-only; fixed standard-data probe; excludes KL and probe refitting',
          'effective_batch': args.batch_size, 'seed': 123,
          'microbatch_size': args.microbatch_size, 'length_grouped': args.length_grouped})
    run = None
    if rank == 0:
        try:
            import wandb
            run = wandb.init(project='training_against_role_probes',
                             name=f'throughput-{os.environ.get("SLURM_JOB_ID", "local")}-{args.context}-mb{args.microbatch_size}',
                             job_type='throughput', config={
                                 'context': args.context, 'effective_batch': args.batch_size,
                                 'layer': 12, 'scope': 'role_only', 'lora_rank': 64,
                                 'microbatch_size': args.microbatch_size,
                                 'length_grouped': args.length_grouped,
                                 'learning_rate': 5e-5, 'fixed_probe': True})
        except Exception as exc:
            emit({'event': 'wandb_unavailable', 'error_type': type(exc).__name__})
    def forward_batch(batch, minimum_length=0):
        # Right padding preserves each trajectory's original position indices.
        length = max(minimum_length, max(len(x['ids']) for x in batch))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        ids = torch.full((len(batch), length), pad_id, device=rank, dtype=torch.long)
        labels = torch.full_like(ids, -1)
        mask = torch.zeros_like(ids)
        for i, example in enumerate(batch):
            n = len(example['ids'])
            ids[i, :n] = torch.tensor(example['ids'], device=rank)
            labels[i, :n] = torch.tensor(example['labels'], device=rank)
            mask[i, :n] = 1
        captured.clear()
        # Padding is excluded from attention and loss. Every row is independent.
        base.model(input_ids=ids, attention_mask=mask, use_cache=False)
        hidden = captured.pop('hidden')
        return hidden, labels

    def loss_for(batch):
        hidden, labels = forward_batch(batch)
        valid = labels >= 0
        logits = F.linear(hidden[valid].float(), weight, bias)
        loss = F.cross_entropy(logits, labels[valid], reduction='sum')
        correct = (logits.argmax(-1) == labels[valid]).sum()
        return loss, correct
    try:
        if args.check_padding:
            # Compare the actual objective and all adapter gradients, not hidden
            # state identity. No optimizer update occurs in this diagnostic.
            check = [dict(x, ids=x['ids'][:n], labels=x['labels'][:n])
                     for x, n in zip(examples, (2048, 1536, 1792, 1024))]
            denominator = sum(sum(y >= 0 for y in x['labels']) for x in check)
            optimizer.zero_grad(set_to_none=True)
            reference_loss = 0.
            for example in check:
                loss, _ = loss_for([example])
                reference_loss += loss.detach().item() / denominator
                (loss / denominator).backward()
            reference = {name: param.grad.detach().cpu().clone()
                         for name, param in model.named_parameters() if param.grad is not None}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(check)
            batched_loss = loss.detach().item() / denominator
            (loss / denominator).backward()
            dot = norm_a = norm_b = difference = 0.
            for name, param in model.named_parameters():
                if name not in reference:
                    if param.grad is not None:
                        raise RuntimeError(f'Gradient presence changed for {name}')
                    continue
                if param.grad is None:
                    raise RuntimeError(f'Gradient missing for {name}')
                left, right = reference.pop(name).double(), param.grad.detach().cpu().double()
                dot += (left * right).sum().item()
                norm_a += left.square().sum().item()
                norm_b += right.square().sum().item()
                difference += (left-right).square().sum().item()
            cosine = dot / (norm_a*norm_b)**.5
            relative_gradient_error = (difference/norm_a)**.5
            relative_loss_error = abs(batched_loss-reference_loss)/abs(reference_loss)
            # Practical calibration gates, not universal numerical guarantees.
            passed = (relative_loss_error < .01 and cosine > .99
                      and relative_gradient_error < .1)
            emit(dict(event='padding_check', reference_loss=reference_loss,
                      batched_loss=batched_loss, relative_loss_error=relative_loss_error,
                      gradient_cosine=cosine, relative_gradient_error=relative_gradient_error,
                      labeled_tokens=denominator, passed=passed))
            del reference, left, right, loss
            optimizer.zero_grad(set_to_none=True)
            if not passed:
                raise RuntimeError('Loss/gradient batching check exceeded calibration tolerances')
        for step in range(1 if args.smoke else args.updates):
            batch = examples[step*args.batch_size:(step+1)*args.batch_size]
            if len(batch) != args.batch_size:
                raise ValueError('Not enough distinct debug examples')
            if args.smoke:
                batch = [dict(x, ids=x['ids'][:128], labels=x['labels'][:128]) for x in batch]
            if args.length_grouped:
                batch = sorted(batch, key=lambda x: len(x['ids']))
            denominator = sum(sum(y >= 0 for y in x['labels']) for x in batch)
            local = batch[rank::world]
            microbatches = [local[i:i+args.microbatch_size]
                            for i in range(0, len(local), args.microbatch_size)]
            padded_tokens = sum(
                len(group) * max(len(x['ids']) for x in group)
                for replica in range(world)
                for sequence in [batch[replica::world]]
                for i in range(0, len(sequence), args.microbatch_size)
                for group in [sequence[i:i+args.microbatch_size]])
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            totals = torch.zeros(2, device=rank, dtype=torch.float64)
            for microbatch in microbatches:
                loss, correct = loss_for(microbatch)
                (loss / denominator).backward()
                totals += torch.stack((loss.detach().double(), correct.double()))
                del loss, correct
            torch.cuda.synchronize()
            forward_backward = time.perf_counter()-started
            # Accumulate token-weighted gradients, then sum across replicas. No
            # averaging here: the global token denominator was already applied.
            for param in parameters:
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError(f'Invalid gradient norm: {gradient_norm}')
            optimizer.step()
            dist.all_reduce(totals)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            record = dict(event='update', step=step, warmup=step == 0, rank=rank,
                          seconds=elapsed, forward_backward_seconds=forward_backward,
                          tokens=sum(len(x['ids']) for x in batch), labeled_tokens=denominator,
                          padded_tokens=padded_tokens, microbatch_size=args.microbatch_size,
                          max_local_length=max(len(x['ids']) for x in local),
                          role_loss=totals[0].item()/denominator,
                          role_accuracy=totals[1].item()/denominator,
                          gradient_norm=gradient_norm.item(),
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          trajectory_ids=[x['id'] for x in batch])
            emit(record)
            if run:
                run.log({k: v for k, v in record.items() if isinstance(v, (float, int))}, step=step)
        # A separate worst-length backward tests feasibility, not average speed.
        if not args.smoke:
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            loss, _ = loss_for([longest] * args.microbatch_size)
            (loss / (args.microbatch_size * sum(y >= 0 for y in longest['labels']))).backward()
            torch.cuda.synchronize()
            emit(dict(event='longest_backward', rank=rank, tokens=len(longest['ids']),
                      microbatch_size=args.microbatch_size,
                      seconds=time.perf_counter()-started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        emit({'event': 'completed', 'rank': rank})
    except Exception as exc:
        emit({'event': 'failed', 'rank': rank, 'error': str(exc), 'type': type(exc).__name__})
        raise
    finally:
        handle.remove()
        if run:
            run.finish()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
