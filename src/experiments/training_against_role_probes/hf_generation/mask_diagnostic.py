"""Reproduce the trained-adapter CUDA crash with controlled kernel selection."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import traceback
import urllib.request
import urllib.error
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from role_probe_training.training.model import RoleModel
from role_probe_training.on_policy.data import sha256
from role_probe_training.on_policy.serving import serve
from role_probe_training.on_policy.objective import reverse_kl_sum
from role_probe_training.generation.masks import eager_decode_masks

ROOT=Path('experiment_reports/training_against_role_probes/on_policy/appworld2700')

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode',choices=['mask-trained','mask-base'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=dict(mode=args.mode,status='started',torch=torch.__version__,
                cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING','0'))
    path=args.output/'result.json'
    def record(**values):
        result.update(values);path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(values),flush=True)
    record()
    runtime=None;started=time.monotonic()
    try:
        torch.cuda.set_device(0);torch.manual_seed(123)
        runtime=RoleModel('.cache/models/gpt-oss-20b',
            'experiment_reports/gptoss_role_probes/gptoss20b_baseline/probes/layer_12.npz',0)
        if not args.mode.endswith('-base'):
            meta=json.loads(Path('experiment_reports/training_against_role_probes/kl_iteration/same_batch/data/post_training/model/export.json').read_text())
            adapter=Path(meta['checkpoint'])/'adapter/adapter_model.safetensors'
            assert sha256(adapter)==meta['adapter_sha256']
            set_peft_model_state_dict(runtime.model,load_file(adapter))
            record(adapter_sha256=meta['adapter_sha256'])
        source=ROOT/'runs/epoch1/rollouts/step00001/rank0/episode1233/generations.jsonl'
        body=json.loads(source.read_text().splitlines()[0])['call']['request']
        body.pop('extra_headers',None);body.update(max_tokens=16,temperature=1,top_p=1,seed=123)
        generation=dict(max_tokens=16,context_limit=32768)
        def post(url,body):
            request=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(request,timeout=900) as response:return json.load(response)
            except urllib.error.HTTPError as error:raise RuntimeError(error.read().decode()) from error
        context=eager_decode_masks()
        timings=[]
        with context:
            with serve(runtime,args.output/'tokens',169,generation) as (url,state):
                post(url+'/episode',dict(id='diagnostic'))
                for index,budget in enumerate((16,128,512)):
                    generation['max_tokens']=budget;body['max_tokens']=budget
                    stamp=time.monotonic();response=post(url+'/v1/chat/completions',body)
                    torch.cuda.synchronize()
                    timings.append(dict(tokens=response['usage']['completion_tokens'],seconds=time.monotonic()-stamp))
                    record(generations=timings)
                if state['errors']:raise RuntimeError('Serving recorded an error')
        # Returning to the unmodified training path must still support backward.
        row=json.loads((args.output/'tokens/requests.jsonl').read_text().splitlines()[0])
        loss=reverse_kl_sum(runtime,row)/len(row['completion_ids'])
        loss.backward();torch.cuda.synchronize()
        gradients=[p.grad for p in runtime.model.parameters() if p.requires_grad and p.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Invalid post-generation gradients')
        record(status='passed',reverse_kl=float(loss.detach()),backward_passed=True)
    except BaseException:
        record(status='failed',traceback=traceback.format_exc());raise
    finally:
        record(wall_seconds=time.monotonic()-started)
        if runtime is not None:runtime.close()

if __name__=='__main__':main()
