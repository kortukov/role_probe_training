"""Whole-history probe CE/accuracy and task success for matched serving attempts."""
import argparse
import json
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from role_probe_training.training.model import RoleModel
from role_probe_training.training.data import read_jsonl,trajectory_examples
from role_probe_training.training.refit import summarize_counts
from role_probe_training.on_policy.data import sha256

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--hf',type=Path,required=True)
    parser.add_argument('--vllm',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    meta=json.loads(Path('experiment_reports/training_against_role_probes/kl_iteration/same_batch/data/post_training/model/export.json').read_text())
    checkpoint=Path(meta['checkpoint']);adapter=checkpoint/'adapter/adapter_model.safetensors'
    if sha256(adapter)!=meta['adapter_sha256']:raise ValueError('Adapter changed')
    for path in (args.hf,args.vllm):
        if json.loads((path/'startup.json').read_text())['adapter_sha256']!=meta['adapter_sha256']:
            raise ValueError('Unmatched serving checkpoints')
    torch.cuda.set_device(0)
    runtime=RoleModel('.cache/models/gpt-oss-20b',str(checkpoint/'layer_12.npz'),0)
    set_peft_model_state_dict(runtime.model,load_file(adapter))
    report=dict(adapter_sha256=meta['adapter_sha256'],probe_sha256=sha256(checkpoint/'layer_12.npz'),
        definitions=dict(probe_ce='Content-token-weighted ground-truth role cross entropy, nats',
            probe_accuracy='Content-token-weighted argmax role accuracy on full histories',
            task_success='Successful tasks / all assigned attempts, including errors and limits'),
        limitations=['Two tasks at each concurrency; diagnostic, not a utility estimate',
            'Five-minute task cap may penalize a slower backend',
            'Both trajectory sets are scored by the same HF checkpoint and fixed refitted probe',
            'HF server serializes concurrent requests; vLLM batches requests',
            'Sampled tokens need not match; HF derives per-call seeds from the requested seed'],backends={})
    try:
        for backend,root in [('hf',args.hf),('vllm',args.vllm)]:
            conditions=[]
            for concurrency in (1,2):
                path=root/f'episodes_c{concurrency}/trajectories.jsonl'
                rows=read_jsonl(path);examples=trajectory_examples(path)
                if len(rows)!=2:raise ValueError('Missing assigned attempts')
                counts=torch.zeros((5,5),dtype=torch.long,device=0);total_loss=0.;scored=0;excluded=[]
                for example in examples:
                    if len(example['ids'])>runtime.base.config.max_position_embeddings:
                        excluded.append(dict(id=example['id'],reason='native_context_limit'));continue
                    loss,matrix=runtime.role_loss([example]);counts+=matrix
                    total_loss+=float(loss);scored+=1;runtime.hidden=None
                summary=summarize_counts(counts)
                success=sum(row.get('scores',{}).get('appworld_success',{}).get('value') in ('C',1) for row in rows)
                conditions.append(dict(concurrency=concurrency,attempts=len(rows),successes=success,
                    task_success=success/len(rows),errors=sum(bool(r['error']) for r in rows),
                    limited=sum(bool(r['limit']) or bool(r['limit_events']) for r in rows),
                    role_scored_attempts=scored,role_coverage=scored/len(rows),unscorable=excluded,
                    probe_ce=total_loss/summary['tokens'] if summary['tokens'] else None,**summary))
            report['backends'][backend]=dict(conditions=conditions,
                micro=json.loads((root/'micro.json').read_text()),
                episodes=json.loads((root/'episodes.json').read_text()))
        report['micro_speed_ratios']=[dict(concurrency=v['concurrency'],
            vllm_over_hf_tokens_per_second=v['output_tokens_per_second']/h['output_tokens_per_second'])
            for h,v in zip(report['backends']['hf']['micro'],report['backends']['vllm']['micro'])]
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:runtime.close()

if __name__=='__main__':main()
