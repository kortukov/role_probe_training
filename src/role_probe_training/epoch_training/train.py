"""Matched epoch training with batch-level or epoch-level rollout refresh."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import shutil

import torch
import torch.distributed as dist
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from role_probe_training.training.data import trajectory_examples, probe_examples
from role_probe_training.training.model import RoleModel
from role_probe_training.training.refit import refit_probe, summarize_counts
from role_probe_training.training.train_same_batch import evaluate, save_checkpoint, sync_gradients
from role_probe_training.epoch_training.data import schedule, select_pool
from role_probe_training.on_policy.data import sha256
from role_probe_training.on_policy.objective import kl_batches, reverse_kl_sums

from role_probe_training.on_policy_recovery.collection import collect


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, help='Pilot boundary; schedule is unchanged')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 1:
        raise ValueError('Epoch comparison requires one GPU')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(hours=6))
    control = dist.new_group(backend='gloo', timeout=timedelta(hours=6))
    torch.manual_seed(config['seed'])
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    batches = schedule(config['trajectories'], config['validation'], config['seed'],
                       epochs=config['epochs'], batch_size=config['batch_size'])
    updates_per_epoch = len(batches) // config['epochs']
    if config['refresh'] not in ('batch', 'epoch'):
        raise ValueError('refresh must be batch or epoch')
    source_paths = [Path(__file__).resolve().relative_to(Path.cwd().resolve())]
    source_paths += list(Path('src/role_probe_training/epoch_training').glob('*.py'))
    source_paths += list(Path('src/role_probe_training/on_policy').glob('*.py'))
    source_paths += list(Path('src/role_probe_training/training').glob('*.py'))
    source_paths += list(Path('src/role_probe_training/on_policy_recovery').glob('*.py'))
    source_paths += [Path('src/role_probe_training/trajectories')/name for name in
                     ('appworld.py', 'replay.py', 'inspect_export.py')]
    source_paths += list(Path('src/role_probe_training/harmony').glob('*.py'))
    source_paths += [
                     Path('src/experiments/eval_role_probes/agentic_eval/appworld/appworld_worker.py'),
                     Path('uv.lock'),
                     Path('src/role_probe_training/post_training/vllm_worker.py')]
    manifest = dict(config=config, world_size=world, updates=len(batches), updates_per_epoch=updates_per_epoch,
        data_sha256={key: sha256(config[key]) for key in
                     ('trajectories', 'validation', 'probe', 'probe_documents')},
        source_sha256={str(path): sha256(path) for path in source_paths},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        image_sha256=sha256(config['image']),
        submodules=subprocess.check_output(['git', 'submodule', 'status'], text=True).strip())
    if rank == 0:
        if not args.resume and (root/'manifest.json').exists():
            raise FileExistsError('Run exists; resume explicitly')
        if not (root/'manifest.json').exists():
            (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            (root/'schedule.json').write_text(json.dumps(batches, indent=2)+'\n')

    def log(event, **values):
        row = dict(event=event, rank=rank, **values)
        print(json.dumps(row), flush=True)
        with (root/f'rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    runtime = RoleModel(config['model'], config['probe'], rank)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=1e-4)
    standard = probe_examples(config['probe_documents'])
    validation = trajectory_examples(config['validation'])
    start_step = 0
    if args.resume:
        state = torch.load(args.resume/'training.pt', map_location='cpu', weights_only=False)
        previous = state['config']
        for key in ('config', 'source_sha256', 'data_sha256', 'world_size', 'image_sha256'):
            if previous[key] != manifest[key]:
                raise ValueError(f'Resume provenance changed: {key}')
        set_peft_model_state_dict(runtime.model, load_file(args.resume/'adapter/adapter_model.safetensors'))
        runtime.probe.load_state_dict(state['probe'])
        optimizer.load_state_dict(state['optimizer'])
        start_step = state['step']
        del state
    run = None
    if rank == 0:
        import wandb
        run = wandb.init(project='training_against_role_probes', name=config['name'],
            id=config['wandb_id'], resume='allow', config=manifest, job_type='epoch-role-training', mode=config.get('wandb_mode', 'offline'))
    stop = args.stop_after or len(batches)
    if not start_step < stop <= len(batches):
        raise ValueError('Invalid stopping step')
    log('ready', start_step=start_step, stop_after=stop, total_updates=len(batches))
    def retain_epoch(epoch, step):
        if rank == 0:
            target = root/'epochs'/f'epoch{epoch:03d}'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copytree(root/'checkpoints'/f'step{step:05d}', target, copy_function=os.link)
        dist.barrier()

    server = None
    try:
        if config.get('serving_backend') == 'colocated_vllm':
            from role_probe_training.on_policy_recovery.colocated import ColocatedServer
            server = ColocatedServer(runtime, root/f'serving/rank{rank}/resume{start_step}')
            server.sleep()
        if start_step == 0:
            log('validation', step=0, metrics=evaluate(runtime, validation))
            save_checkpoint(runtime, optimizer, root/'checkpoints', 0, manifest, 1)
            retain_epoch(0, 0)
        for index in range(start_step, stop):
            step = index+1
            started = time.perf_counter()
            error = None
            epoch_index = index // updates_per_epoch
            epoch_start = epoch_index * updates_per_epoch
            pool_path = root/'pools'/f'epoch{epoch_index+1:03d}.pt'
            try:
                refresh = config['refresh'] == 'batch' or index == epoch_start
                if refresh:
                    if server is not None:
                        log('serving_refresh', step=step, refresh=server.synchronize(step-1))
                    episodes = (batches[index] if config['refresh'] == 'batch' else
                                [e for b in batches[epoch_start:epoch_start+updates_per_epoch] for e in b])
                    examples, requests, records = collect(runtime, config,
                        episodes, step, root, rank,
                        sampler=server.sample if server is not None else None)
                    if server is not None:
                        log('serving_distribution', step=step,
                            comparison=server.compare_distribution(requests[0]['prompt_ids']))
                    if config['refresh'] == 'epoch':
                        pool_path.parent.mkdir(exist_ok=True)
                        temporary = pool_path.with_suffix('.tmp')
                        torch.save(dict(examples=examples, requests=requests, records=records,
                                        policy_step=step-1), temporary)
                        temporary.replace(pool_path)
                if config['refresh'] == 'epoch':
                    pool = torch.load(pool_path, map_location='cpu', weights_only=False)
                    if pool['policy_step'] != epoch_start:
                        raise ValueError('Epoch pool came from the wrong policy checkpoint')
                    examples, requests, records = select_pool(pool, batches[index])
                    del pool
            except Exception:
                error = traceback.format_exc()
                log('collection_failed', step=step, error=error)
            finally:
                if server is not None and not server.sleeping:
                    server.sleep()
            failures = torch.tensor(int(error is not None))
            dist.all_reduce(failures, group=control)
            if failures.item():
                raise RuntimeError('Collection failed on a rank; no update or automatic reroll')
            collection_seconds = time.perf_counter()-started
            log('collection_complete', step=step, local_episodes=len(records),
                local_requests=len(requests), seconds=collection_seconds)
            denominators = torch.tensor([
                sum(sum(y >= 0 for y in x['labels']) for x in examples),
                sum(len(x['completion_ids']) for x in requests)], device=rank, dtype=torch.float64)
            dist.all_reduce(denominators)
            if (denominators <= 0).any():
                raise ValueError('Empty role or KL objective')
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(2, device=rank, dtype=torch.float64)
            counts = torch.zeros((5,5), device=rank, dtype=torch.long)
            torch.cuda.reset_peak_memory_stats()
            for example in examples:
                loss, confusion = runtime.role_loss([example])
                (loss/denominators[0]).backward()
                totals[0] += loss.detach().double()
                counts += confusion
                runtime.hidden = None
                del loss, confusion
            done = 0
            for batch in kl_batches(requests, config.get('kl_batch_tokens')):
                loss = reverse_kl_sums(runtime, batch)
                (config['kl_coefficient']*loss/denominators[1]).backward()
                totals[1] += loss.detach().double()
                runtime.hidden = None
                del loss
                done += len(batch)
                if done//32 > (done-len(batch))//32:
                    log('kl_progress',step=step,requests_done=done,
                        requests_total=len(requests),seconds=time.perf_counter()-started)
            sync_gradients(parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(grad_norm):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            dist.all_reduce(totals)
            dist.all_reduce(counts)
            peak = torch.cuda.max_memory_allocated()/2**30
            probe = refit_probe(runtime, standard)
            successes = sum(row.get('scores', {}).get('appworld_success', {}).get('value') in ('C', 1)
                            for row in records)
            attempts = torch.tensor([successes, len(records)], device=rank)
            dist.all_reduce(attempts)
            record = dict(step=step, epoch=epoch_index+1, rollout_policy_step=(step-1 if config['refresh']=='batch' else epoch_start), role_loss=(totals[0]/denominators[0]).item(),
                reverse_kl=(totals[1]/denominators[1]).item(),
                role_accuracy=summarize_counts(counts)['accuracy'],
                collection_successes=int(attempts[0]), collection_attempts=int(attempts[1]),
                gradient_norm=grad_norm.item(), peak_update_gib=peak,
                collection_seconds=collection_seconds, refit_seconds=probe['seconds'],
                refit_iterations=probe['iterations'], probe_training_tokens=probe['training_tokens'],
                probe_heldout_accuracy=probe['accuracy'], total_seconds=time.perf_counter()-started)
            log('update', **record)
            if run:
                run.log(record, step=step)
            if step % config['eval_interval'] == 0 or step == stop:
                log('validation', step=step, metrics=evaluate(runtime, validation), standard_probe=probe)
            # Every update is a recoverable policy boundary, including the pilot.
            save_checkpoint(runtime, optimizer, root/'checkpoints', step, manifest, 1)
            if step % updates_per_epoch == 0:
                retain_epoch(step // updates_per_epoch, step)
                log('epoch_complete', epoch=step // updates_per_epoch, step=step)
        log('completed' if stop == len(batches) else 'pilot_completed', updates=stop)
    finally:
        if server is not None:
            server.close()
        runtime.close()
        if run:
            run.finish()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
