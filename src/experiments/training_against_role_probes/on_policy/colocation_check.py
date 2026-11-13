"""Two-GPU colocation gate: fresh episodes, exact KL, update, full refit, refresh.

Two episodes form a diagnostic batch only. Production retains batch16/2700.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from role_probe_training.training.model import RoleModel
from role_probe_training.training.data import probe_examples
from role_probe_training.training.refit import refit_probe
from role_probe_training.training.train_same_batch import sync_gradients
from role_probe_training.on_policy.colocated import ColocatedServer
from role_probe_training.on_policy.data import schedule
from role_probe_training.on_policy.objective import reverse_kl_sum
from role_probe_training.on_policy.train import collect


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    config = json.loads(Path('src/experiments/training_against_role_probes/on_policy/config.json').read_text())
    torch.manual_seed(config['seed'])
    runtime = RoleModel(config['model'],config['probe'],rank)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=1e-4)
    args.output.mkdir(parents=True,exist_ok=True)
    log = args.output/f'rank{rank}.jsonl'
    def record(event, **data):
        row = dict(event=event,rank=rank,**data)
        print(json.dumps(row),flush=True)
        with log.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    def memory():
        free,total = torch.cuda.mem_get_info()
        return dict(free_gib=free/2**30,total_gib=total/2**30,
                    trainer_allocated_gib=torch.cuda.memory_allocated()/2**30)
    server = None
    try:
        record('trainer_loaded',**memory())
        server = ColocatedServer(runtime,args.output/f'server{rank}')
        record('server_loaded',**memory())
        refresh = server.synchronize(0)
        record('policy_zero_ready',refresh=refresh,**memory())
        batches = schedule(config['trajectories'],config['validation'],config['seed'])
        started=time.monotonic()
        examples,requests,records=collect(runtime,config,[batches[0][rank]],1,args.output,rank,
                                          sampler=server.sample)
        record('episode_complete',requests=len(requests),seconds=time.monotonic()-started)
        record('distribution_zero',comparison=server.compare_distribution(requests[0]['prompt_ids']))
        server.sleep()
        record('server_sleeping',**memory())
        denominators=torch.tensor([sum(sum(y>=0 for y in x['labels']) for x in examples),
            sum(len(x['completion_ids']) for x in requests)],device=rank,dtype=torch.float64)
        dist.all_reduce(denominators)
        totals=torch.zeros(2,device=rank,dtype=torch.float64)
        torch.cuda.reset_peak_memory_stats()
        for example in examples:
            loss,_=runtime.role_loss([example])
            (loss/denominators[0]).backward()
            totals[0]+=loss.detach()
            runtime.hidden=None
            del loss
        for request in requests:
            loss=reverse_kl_sum(runtime,request)
            (config['kl_coefficient']*loss/denominators[1]).backward()
            totals[1]+=loss.detach()
            runtime.hidden=None
            del loss
        sync_gradients(parameters)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or norm<=0:
            raise ValueError('Invalid gradient')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(totals)
        if abs(float(totals[1]/denominators[1]))>1e-5:
            raise ValueError('Initial policy KL must be zero')
        record('updated',gradient_norm=float(norm),reverse_kl=float(totals[1]/denominators[1]),
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**memory())
        probe=refit_probe(runtime,probe_examples(config['probe_documents']))
        record('refitted',probe=probe)
        refresh=server.synchronize(1)
        record('policy_one_ready',refresh=refresh,**memory())
        record('distribution_one',comparison=server.compare_distribution(requests[0]['prompt_ids']))
        # A new episode must use the refreshed model; retain its full trace too.
        examples2,requests2,_=collect(runtime,config,[batches[1][rank]],2,args.output,rank,
                                      sampler=server.sample)
        server.sleep()
        with torch.no_grad():
            kl=reverse_kl_sum(runtime,requests2[0])/len(requests2[0]['completion_ids'])
        if not torch.isfinite(kl) or kl< -1e-6:
            raise ValueError('Post-update KL invalid')
        record('passed',post_update_kl=float(kl),fresh_requests=len(requests2))
        dist.barrier()
        if rank==0:
            (args.output/'passed.json').write_text(json.dumps(dict(two_gpu_colocation=True,
                full_episode_per_rank_before_and_after=True,
                production_batch_size_validated=False,checkpoint_resume_validated=False),indent=2)+'\n')
    finally:
        if server is not None:server.close()
        runtime.close()
        dist.destroy_process_group()

if __name__=='__main__':main()
