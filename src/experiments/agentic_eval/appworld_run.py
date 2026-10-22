"""Collect split-separated AppWorld trajectories with one shared inference server."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, get_model
from role_probe_training.trajectories.appworld import appworld_task
from role_probe_training.trajectories.inspect_export import export_logs


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--split', choices=['train','dev','test_normal','test_challenge'], required=True)
    p.add_argument('--samples', type=int)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--image', default='.cache/appworld/appworld.sif')
    p.add_argument('--data', default='.cache/appworld/data')
    args = p.parse_args()
    ids = sorted((Path(args.data)/'datasets'/f'{args.split}.txt').read_text().splitlines())
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.samples] if args.samples else ids
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise ValueError('Use a fresh output directory')
    manifest = {'benchmark':'appworld_'+args.split, 'sample_ids':ids,
        'settings':vars(args)|{'output':str(args.output)},
        'model':'openai/gpt-oss-20b', 'model_revision':'6cee5e81ee83917806bbde320786a8fb61efebee',
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'container_sha256':hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
        'data_source':'https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    model = get_model('vllm/openai/gpt-oss-20b',base_url=args.base_url,api_key='local',
        config=GenerateConfig(temperature=1,seed=args.seed,max_tokens=args.max_tokens,
                              reasoning_effort='high',max_retries=1,timeout=300,
                              max_connections=args.concurrency))
    task = appworld_task(ids,args.split,args.image,args.data,args.output/'worlds',
                         args.output.name,max_steps=args.max_steps)
    started=time.perf_counter()
    logs=eval(task,model=model,max_samples=args.concurrency,max_connections=args.concurrency,
              time_limit=1800,token_limit=500000,fail_on_error=False,
              log_dir=str(args.output/'logs'),log_model_api=True,display='plain')
    elapsed=time.perf_counter()-started
    export_logs([l.location for l in logs],args.output,'appworld_'+args.split,manifest)
    timings=[]
    for log in logs:
        for s in read_eval_log(log.location).samples:
            timings.append({'sample_id':s.id, **(s.metadata or {}).get('appworld',{}),
                            'error':bool(s.error)})
    (args.output/'timing.json').write_text(json.dumps({'split':args.split,
        'wall_seconds':elapsed,'tasks':len(ids),'concurrency':args.concurrency,
        'trajectories_per_hour':len(timings)*3600/elapsed,'samples':timings},indent=2)+'\n')
    if len(timings)!=len(ids) or any(l.status!='success' for l in logs):
        raise SystemExit('Incomplete evaluation; inspect logs')


if __name__=='__main__':
    main()
