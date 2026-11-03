"""Replay the failed pilot input with synchronized CUDA errors, outside training.

Each attention/thread variant runs in its own process/allocation because an
illegal CUDA access poisons the process. Diagnostic samples never enter training.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
import urllib.error
import urllib.request

import torch
from role_probe_training.training.data import trajectory_examples
from role_probe_training.training.model import RoleModel
from role_probe_training.on_policy.objective import reverse_kl_sum
from role_probe_training.on_policy.serving import serve, template_messages

ROOT = Path('experiment_reports/training_against_role_probes/on_policy/appworld2700')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=['flex-thread', 'flex-main', 'eager-thread'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path('src/experiments/training_against_role_probes/on_policy/config.json').read_text())
    source = ROOT/'runs/epoch1/rollouts/step00001/rank0/episode1233/generations.jsonl'
    saved = json.loads(source.read_text().splitlines()[0])['call']['request']
    saved.pop('extra_headers', None)
    saved['max_tokens'] = 768
    result = dict(mode=args.mode, status='started', source=str(source), diagnostic_only=True,
                  torch=torch.__version__, cuda_launch_blocking=True, max_tokens=768)
    path = args.output/'result.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    started = time.perf_counter()
    runtime = None
    try:
        torch.cuda.set_device(0)
        torch.manual_seed(123)
        runtime = RoleModel(config['model'], config['probe'], 0)
        result['gpu'] = torch.cuda.get_device_name()
        # Match the original pilot's validation-before-generation sequence.
        validation = trajectory_examples(runtime.tokenizer, config['validation'])
        with torch.no_grad():
            for example in validation[::2]:
                runtime.role_loss([example])
        torch.cuda.synchronize()
        result['validation_passed'] = True
        path.write_text(json.dumps(result, indent=2)+'\n')
        if args.mode == 'eager-thread':
            # Short diagnostic only. Never enable quadratic attention silently
            # for full-history training or production generation.
            runtime.base.config._attn_implementation = 'eager'
        generation = dict(max_tokens=768, context_limit=32768)
        if args.mode.endswith('thread'):
            with serve(runtime, args.output, 0, generation) as (url, state):
                request = urllib.request.Request(url+'/episode', data=b'{"id":"diagnostic"}',
                    headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=30):
                    pass
                request = urllib.request.Request(url+'/v1/chat/completions',
                    data=json.dumps(saved).encode(), headers={'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=900) as response:
                        response_body = json.load(response)
                    (args.output/'response.json').write_text(json.dumps(response_body, indent=2)+'\n')
                except urllib.error.HTTPError as error:
                    raise RuntimeError(error.read().decode()) from error
                assert not state['errors']
            row = json.loads((args.output/'requests.jsonl').read_text().splitlines()[0])
        else:
            prompt = runtime.tokenizer.apply_chat_template(template_messages(saved['messages']),
                tools=saved.get('tools'), tokenize=True, add_generation_prompt=True, reasoning_effort='high')
            runtime.model.gradient_checkpointing_disable()
            ids = torch.tensor([prompt], device=0)
            stops = [runtime.tokenizer.convert_tokens_to_ids(x) for x in ('<|return|>', '<|call|>')]
            with torch.random.fork_rng(devices=[0]), torch.no_grad():
                torch.manual_seed(saved['seed'])
                output = runtime.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=True, temperature=1., top_p=1., top_k=0, repetition_penalty=1.,
                    max_new_tokens=768, eos_token_id=stops, pad_token_id=runtime.pad_id, use_cache=True)
            row = dict(prompt_ids=prompt, completion_ids=output[0,len(prompt):].tolist())
            (args.output/'requests.jsonl').write_text(json.dumps(row)+'\n')
            runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.cuda.synchronize()
        result.update(generation_passed=True, generated_tokens=len(row['completion_ids']),
                      generation_seconds=time.perf_counter()-started)
        path.write_text(json.dumps(result, indent=2)+'\n')
        runtime.base.config._attn_implementation = 'flex_attention'
        runtime.hidden = None
        loss = reverse_kl_sum(runtime, row)/len(row['completion_ids'])
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss) and abs(loss.item()) < 1e-5
        gradients = [p.grad for p in runtime.model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        result.update(status='passed', initial_reverse_kl=loss.item(), backward_passed=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        path.write_text(json.dumps(result, indent=2)+'\n')
        if runtime is not None:
            runtime.close()


if __name__ == '__main__':
    main()
