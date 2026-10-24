"""Isolated same-batch KL trainer; the active fixed-pool trainer stays unchanged."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import torch
import torch.distributed as dist
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from .data import trajectory_examples, probe_examples, kl_examples, read_jsonl
from .model import RoleModel
from .objectives import kl_sum
from .refit import refit_probe, summarize_counts


def sync_gradients(parameters):
    for param in parameters:
        # KL touches all adapted layers; missing gradients should be consistent.
        present=torch.tensor(int(param.grad is not None),device=param.device)
        dist.all_reduce(present)
        if present.item()==0: continue
        if param.grad is None: param.grad=torch.zeros_like(param)
        dist.all_reduce(param.grad)


@torch.no_grad()
def evaluate(runtime, examples):
    rank=dist.get_rank(); world=dist.get_world_size()
    confusion=torch.zeros((5,5),device=runtime.device,dtype=torch.long)
    for example in examples[rank::world]:
        _,counts=runtime.role_loss([example]); confusion+=counts
    dist.all_reduce(confusion)
    return summarize_counts(confusion)


def save_checkpoint(runtime, optimizer, root, step, config, microbatch):
    if dist.get_rank()==0:
        target=root/f'step{step:05d}'; temporary=root/f'.step{step:05d}.tmp'
        if temporary.exists(): shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        runtime.model.save_pretrained(temporary/'adapter')
        runtime.save_probe(temporary/'layer_12.npz',step)
        torch.save(dict(optimizer=optimizer.state_dict(),probe=runtime.probe.state_dict(),
                        step=step,microbatch=microbatch,config=config),temporary/'training.pt')
        temporary.rename(target)
        pointer=root/'latest.json.tmp'; pointer.write_text(json.dumps({'path':str(target)}))
        pointer.replace(root/'latest.json')
        # Keep two complete restart states; never remove the only valid checkpoint.
        for stale in sorted(root.glob('step[0-9]*'))[:-2]: shutil.rmtree(stale)
    dist.barrier()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    a=p.parse_args(); config=json.loads(a.config.read_text())
    rank=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    control=dist.new_group(backend='gloo',timeout=timedelta(minutes=60))
    torch.manual_seed(config.get('seed',123))
    root=Path(config['output']); root.mkdir(parents=True,exist_ok=True)
    def log(event,**values):
        row={'event':event,'rank':rank,**values}
        print(json.dumps(row),flush=True)
        with (root/f'rank{rank}.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    runtime=RoleModel(config['model'],config['probe'],rank)
    parameters=[x for x in runtime.model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config.get('learning_rate',5e-5),weight_decay=1e-4)
    training=trajectory_examples(runtime.tokenizer,config['trajectories'])
    validation=trajectory_examples(runtime.tokenizer,config['validation'])
    training_tasks={r['sample_id'] for r in read_jsonl(config['trajectories'])}
    validation_tasks={r['sample_id'] for r in read_jsonl(config['validation'])}
    if training_tasks & validation_tasks: raise ValueError('Training and validation tasks overlap')
    standard=probe_examples(runtime.tokenizer,config['probe_documents'])
    anchors=kl_examples(config['kl_anchors'])
    if any(len(x['ids'])>runtime.base.config.max_position_embeddings for x in anchors):
        raise ValueError('KL history exceeds native context; no truncation is allowed')
    if config.get('kl_sampling') != 'same_role_batch':
        raise ValueError('This isolated trainer requires same_role_batch KL')
    anchor_by_id={x['id']:x for x in anchors}
    if len(anchor_by_id)!=len(anchors) or set(anchor_by_id)!={x['id'] for x in training}:
        raise ValueError('Same-batch KL must cover exactly the role-training corpus')
    if any(anchor_by_id[x['id']]['ids']!=x['ids'] for x in training):
        raise ValueError('KL and role histories must have identical token IDs')
    if len(training)!=config['expected_trajectories']:
        raise ValueError('Unexpected training corpus size')
    manifest={**config,'world_size':world,'trainable_parameters':sum(x.numel() for x in parameters),
              'sha256':{k:hashlib.sha256(Path(config[k]).read_bytes()).hexdigest() for k in
                        ('trajectories','validation','probe_documents','kl_anchors','probe')}}
    manifest['gpu']=torch.cuda.get_device_name()
    manifest['torch']=torch.__version__
    manifest['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    if rank==0: (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    batches=[]; generator=random.Random(config.get('seed',123))
    for epoch in range(config.get('epochs',1)):
        order=training.copy(); generator.shuffle(order)
        batches.extend([sorted(order[i:i+16],key=lambda x:len(x['ids'])) for i in range(0,len(order),16)])
    if config.get('max_updates'): batches=batches[:config['max_updates']]
    start_step=0; microbatch=config.get('microbatch',2)
    if a.resume:
        state=torch.load(a.resume/'training.pt',map_location='cpu',weights_only=False)
        for key in ('trajectories','probe_documents','kl_anchors'):
            if state['config'][key]!=config[key]: raise ValueError('Resume data changed')
        if state['config'].get('sha256')!=manifest['sha256']: raise ValueError('Resume artifact hashes changed')
        weights=load_file(a.resume/'adapter/adapter_model.safetensors')
        set_peft_model_state_dict(runtime.model,weights)
        runtime.probe.load_state_dict(state['probe']); optimizer.load_state_dict(state['optimizer'])
        start_step=state['step']; microbatch=state['microbatch']
        del state,weights
    run=None
    if rank==0:
        import wandb
        run=wandb.init(project='training_against_role_probes',name=config['name'],config=manifest,
                       id=config.get('wandb_id'),resume='allow',job_type='training')
    log('ready',updates=len(batches),start_step=start_step,trajectories=len(training),
        kl_pairs=len(anchors),probe_passage_roles=len(standard))
    try:
        if start_step==0:
            metrics=evaluate(runtime,validation); log('validation',step=0,metrics=metrics)
            if run: run.log({'validation/accuracy':metrics['accuracy'],
                             'validation/macro_recall':metrics['macro_recall']},step=0)
            save_checkpoint(runtime,optimizer,root/'checkpoints',0,manifest,microbatch)
        for index,batch in enumerate(batches):
            if index<start_step: continue
            step=index+1; start=time.perf_counter()
            kl_batch=[anchor_by_id[x['id']] for x in batch]
            role_denominator=sum(sum(y>=0 for y in x['labels']) for x in batch)
            kl_denominator=sum(sum(x['completion_mask']) for x in kl_batch)
            if kl_denominator==0: raise ValueError('KL batch has no scored tokens')
            local=batch[rank::world]; local_kl=kl_batch[rank::world]
            injected_oom=False
            while True:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                confusion=torch.zeros((5,5),device=rank,dtype=torch.long)
                totals=torch.zeros(2,device=rank,dtype=torch.float64)
                error=None; role_seconds=kl_seconds=0.
                try:
                    phase=time.perf_counter()
                    for j in range(0,len(local),microbatch):
                        loss,counts=runtime.role_loss(local[j:j+microbatch])
                        (loss/role_denominator).backward()
                        runtime.hidden=None
                        if config.get('test_oom_recovery') and rank==1 and not injected_oom and microbatch>1:
                            injected_oom=True
                            raise torch.cuda.OutOfMemoryError('Injected smoke-test allocation failure')
                        confusion+=counts; totals[0]+=loss.detach().double()
                        del loss,counts
                    torch.cuda.synchronize(); role_seconds=time.perf_counter()-phase
                    phase=time.perf_counter()
                    for example in local_kl:
                        if not any(example['completion_mask']): continue
                        loss=kl_sum(runtime,[example])
                        (config.get('kl_coefficient',16)*loss/kl_denominator).backward()
                        runtime.hidden=None
                        totals[1]+=loss.detach().double(); del loss
                    torch.cuda.synchronize(); kl_seconds=time.perf_counter()-phase
                except torch.cuda.OutOfMemoryError:
                    error='oom'
                    runtime.hidden=None
                    if 'loss' in locals(): del loss
                failure=torch.tensor(int(error is not None),dtype=torch.int32)
                dist.all_reduce(failure,group=control)
                if failure.item()==0: break
                optimizer.zero_grad(set_to_none=True); runtime.hidden=None
                torch.cuda.empty_cache()
                if microbatch==1:
                    raise RuntimeError('OOM at microbatch1; resume last checkpoint after changing memory strategy')
                microbatch=1; log('retry_update',step=step,microbatch=microbatch)
            sync_gradients(parameters)
            grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(grad_norm): raise RuntimeError('Nonfinite model gradient')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            dist.all_reduce(totals); dist.all_reduce(confusion)
            torch.cuda.synchronize(); update_seconds=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated()/2**30
            # Freeze model computation and refresh probes on standard data only.
            probe_metrics=refit_probe(runtime,standard,steps=config.get('refit_steps',2000),seed=123+step)
            record=dict(step=step,role_loss=totals[0].item()/role_denominator,
                kl=totals[1].item()/kl_denominator,role_accuracy=summarize_counts(confusion)['accuracy'],
                gradient_norm=grad_norm.item(),microbatch=microbatch,update_seconds=update_seconds,
                role_seconds=role_seconds,kl_seconds=kl_seconds,peak_update_gib=peak,
                refit_seconds=probe_metrics['seconds'],probe_heldout_accuracy=probe_metrics['accuracy'],
                total_seconds=time.perf_counter()-start)
            log('update',**record)
            if run: run.log(record,step=step)
            if step%config.get('eval_interval',5)==0 or step==len(batches):
                validation_metrics=evaluate(runtime,validation)
                log('validation',step=step,metrics=validation_metrics,standard_probe=probe_metrics)
                if run: run.log({'validation/accuracy':validation_metrics['accuracy'],
                                 'validation/macro_recall':validation_metrics['macro_recall']},step=step)
            if step%config.get('checkpoint_interval',10)==0 or step==len(batches):
                save_checkpoint(runtime,optimizer,root/'checkpoints',step,manifest,microbatch)
        log('completed',updates=len(batches))
    finally:
        runtime.close()
        if run: run.finish()
        dist.destroy_process_group()

if __name__=='__main__': main()
