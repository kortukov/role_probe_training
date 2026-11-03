"""Run the matched benchmark with the original HF/PEFT adapter snapshot."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from role_probe_training.training.model import RoleModel
from role_probe_training.on_policy.data import sha256
from role_probe_training.on_policy.serving import serve


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export = Path('experiment_reports/training_against_role_probes/kl_iteration/same_batch/data/post_training/model/export.json')
    metadata = json.loads(export.read_text())
    adapter = Path(metadata['checkpoint'])/'adapter/adapter_model.safetensors'
    assert sha256(adapter)==metadata['adapter_sha256']
    torch.cuda.set_device(0)
    started = time.perf_counter()
    runtime = RoleModel('.cache/models/gpt-oss-20b',
        'experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz',0)
    set_peft_model_state_dict(runtime.model, load_file(adapter))
    torch.cuda.synchronize()
    (args.output/'startup.json').write_text(json.dumps(dict(backend='HF/PEFT',
        load_seconds=time.perf_counter()-started,adapter_sha256=metadata['adapter_sha256'],
        torch=torch.__version__,gpu=torch.cuda.get_device_name()),indent=2)+'\n')
    generation = dict(max_tokens=16,context_limit=32768)
    try:
        with serve(runtime,args.output/'tokens',169,generation) as (url,state):
            request = urllib.request.Request(url+'/episode',data=b'{"id":"benchmark"}',
                                            headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request):
                pass
            for phase,tokens in [('warmup',16),('micro',128),('episodes',8192)]:
                generation['max_tokens']=tokens
                subprocess.run(['.tools/bin/uv','run','--project','src/experiments/agentic_eval/swe',
                    '--frozen','--no-sync','python',
                    'src/experiments/training_against_role_probes/on_policy/benchmark_client.py',
                    '--url',url,'--model','on-policy-role-model','--output',str(args.output),
                    '--phase',phase],check=True,timeout=2400)
                if state['errors']:
                    raise RuntimeError('HF benchmark failed; no speedup comparison is valid')
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
