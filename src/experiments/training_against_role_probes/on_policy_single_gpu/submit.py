"""Replace the two pending 270 runs and reconnect existing evaluation DAGs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE=Path('experiment_reports/training_against_role_probes/on_policy')
CODE=Path('src/experiments/training_against_role_probes/on_policy_single_gpu')
SCRIPT=Path('src/experiments/training_against_role_probes/on_policy_execution/train.sbatch')

def call(command, **kwargs):
    return subprocess.check_output(command,text=True,**kwargs).strip()

def state(job):
    return call(['squeue','-h','-j',job,'-o','%T'])


def main():
    runs=[('appworld270_batch64','1816549',5),('appworld270_batch16','1816551',17)]
    for name,old,_ in runs:
        assert state(old)=='PENDING', (name,state(old))
        assert not (BASE/name/'single_gpu_jobs.json').exists()
        evaluation=json.loads((BASE/name/'evaluation_jobs.json').read_text())
        assert state(evaluation['jobs']['export']['job_id'])=='PENDING'
    for name,old,stop in runs:
        root=BASE/name
        migration=json.loads((CODE/f'{name}_migration.json').read_text())
        checkpoint=Path(migration['new_config']['output'])/f'checkpoints/step{stop:05d}'
        evaluation=json.loads((root/'evaluation_jobs.json').read_text())
        oldexport=evaluation['jobs']['export']['job_id']
        # Hold the old job to avoid duplicate execution during the replacement.
        subprocess.run(['scontrol','hold',old],check=True)
        env=dict(os.environ,VARIANT_CONFIG=str(CODE/f'{name}.json'),VARIANT_REPORT=str(root))
        command=['sbatch','--parsable','--hold','--export=ALL','--gpus=1',
                 f'--job-name=role-onpolicy-1gpu-{name}',f'--output={root}/logs/%x-%j.out',
                 str(SCRIPT),'--resume',migration['checkpoint'],
                 '--migration',str(CODE/f'{name}_migration.json'),'--stop-after',str(stop)]
        job=call(command,env=env).split(';')[0]
        record=dict(training_job=job,old_training_job=old,command=command,
                    world_size=1,resume_step=migration['step'],final_step=stop,checkpoint=str(checkpoint),
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path('src/role_probe_training/on_policy_training/train.py'),SCRIPT,CODE/f'{name}.json',CODE/f'{name}_migration.json']})
        (root/'single_gpu_jobs.json').write_text(json.dumps(record,indent=2)+'\n')
        export_command=[f'--dependency=afterok:{job}' if a.startswith('--dependency=') else a
                        for a in evaluation['jobs']['export']['command'] if not a.startswith('--nice=')]
        export_command.insert(1,'--hold')
        index=next(i for i,a in enumerate(export_command) if a.endswith('/post_training/export.sbatch'))
        export_command[index+1]=str(checkpoint)
        export=call(export_command).split(';')[0]
        record['export_job']=export
        (root/'single_gpu_jobs.json').write_text(json.dumps(record,indent=2)+'\n')
        (root/'evaluation_jobs_before_single_gpu.json').write_text(json.dumps(evaluation,indent=2)+'\n')
        evaluation['jobs']['export'].update(job_id=export,command=export_command)
        for stage,item in evaluation['jobs'].items():
            if stage=='export':continue
            olddep=next(a for a in item['command'] if a.startswith('--dependency='))
            newdep=':'.join(export if token==oldexport else token for token in olddep.split(':'))
            if newdep!=olddep:
                subprocess.run(['scontrol','update',f'JobId={item["job_id"]}',newdep.replace('--dependency=','Dependency=')],check=True)
                item['command']=[newdep if a==olddep else a for a in item['command']]
        evaluation.update(training_job=job,checkpoint=str(checkpoint),parser_training_manifest='single_gpu_jobs.json')
        (root/'evaluation_jobs.json').write_text(json.dumps(evaluation,indent=2)+'\n')
        subprocess.run(['scancel',old,oldexport],check=True)
        subprocess.run(['scontrol','release',export],check=True)
        subprocess.run(['scontrol','release',job],check=True)
        print(name,'training',job,'export',export,flush=True)


if __name__=='__main__':
    main()
