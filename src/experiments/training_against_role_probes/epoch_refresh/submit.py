"""Plan or submit the paired ten-epoch training and retail evaluation DAG."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

CODE = Path('src/experiments/training_against_role_probes/epoch_refresh')
BASE = Path('experiment_reports/training_against_role_probes')
EXCLUDE = 'vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if args.submit:
        if not (Path.cwd()/'FROZEN_SOURCE.json').is_file():
            raise ValueError('Submit only from a reviewed source snapshot with FROZEN_SOURCE.json')
        for name in ('AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT'):
            if not os.environ.get(name):
                raise ValueError(f'Missing retail credential: {name}')
        for mode in ('on_policy','off_policy'):
            if (BASE/f'{mode}_epochs/jobs.json').exists():
                raise FileExistsError(f'{mode}: inspect existing jobs before any retry')
    os.environ['PYTHONPATH'] = str(Path.cwd()/'src')
    for mode in ('on_policy','off_policy'):
        root=BASE/f'{mode}_epochs'
        config=CODE/f'{mode}.json'
        c=json.loads(config.read_text())
        record=dict(mode=mode,source_snapshot=str(Path.cwd()),config=c,jobs={},
                    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    submodules=subprocess.check_output(['git','submodule','status'],text=True).strip(),
                    plan=dict(tasks_per_epoch=90,updates_per_epoch=6,epochs=10,evaluated_epochs=list(range(11))))
        if args.submit:
            (root/'logs').mkdir(exist_ok=True)
            (root/'checkpoints').mkdir(exist_ok=True)
        def submit(name,script,arguments=(),dependencies=(),env=None):
            path=CODE/script
            command=['sbatch','--parsable','--export=ALL',f'--chdir={Path.cwd()}',
                     f'--exclude={EXCLUDE}',f'--output={root}/logs/%x-%j.out']
            if dependencies:
                command += ['--dependency=afterok:'+':'.join(dependencies)]
            command += [str(path),*map(str,arguments)]
            job=(subprocess.check_output(command,text=True,env=dict(os.environ,**(env or {}))).strip().split(';')[0]
                 if args.submit else name)
            record['jobs'][name]=dict(job_id=job,command=command,script_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if args.submit:
                (root/'jobs.json').write_text(json.dumps(record,indent=2)+'\n')
            print(mode,name,job,flush=True)
            return job
        previous=None
        for epoch in range(1,11):
            arguments=['--stop-after',epoch*6]
            if epoch>1:
                arguments += ['--resume',Path(c['output'])/'checkpoints'/f'step{(epoch-1)*6:05d}']
            training=submit(f'train_epoch{epoch:03d}','train.sbatch',arguments,
                            [previous] if previous else [],
                            dict(VARIANT_CONFIG=str(config),VARIANT_REPORT=str(root)))
            # Evaluation depends on the immutable retained epoch checkpoint. Further
            # training uses latest recovery states and does not wait for evaluation.
            for evaluated in ([0,1] if epoch==1 else [epoch]):
                checkpoint=Path(c['output'])/'epochs'/f'epoch{evaluated:03d}'
                evaluation=submit(f'evaluate_epoch{evaluated:03d}','evaluate.sbatch',
                                  [root/'checkpoints'/f'epoch{evaluated:03d}',checkpoint],[training])
                submit(f'publish_epoch{evaluated:03d}','publish.sbatch',[root],[evaluation])
            previous=training
        if not args.submit:
            print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
