"""Submit audited checkpoint recoveries and reconnect existing evaluations."""
import hashlib,json,os,subprocess
from pathlib import Path

BASE=Path('experiment_reports/training_against_role_probes/on_policy')
CODE=Path('src/experiments/training_against_role_probes/on_policy_harmony')

def call(command):return subprocess.check_output(command,text=True).strip()

for name in ('appworld270_batch64','appworld270_batch16','appworld2700'):
    if (BASE/name/'harmony_jobs.json').exists():raise FileExistsError(name+' already submitted')
# Only replace the pending remnants of the failed long-run chain.
for job in ('1798634','1798635','1798636','1798637','1798638','1798639'):
    state=call(['squeue','-h','-j',job,'-o','%T'])
    if state=='PENDING':subprocess.run(['scancel',job],check=True)
    elif state:raise RuntimeError(f'Unexpected active state: {job} {state}')

for name,nice,ends in [('appworld270_batch64',0,[5]),('appworld270_batch16',100,[17]),('appworld2700',1000,[26,50,74,98,122,146,169])]:
    root=BASE/name; config=CODE/f'{name}.json';migration=CODE/f'{name}_migration.json'
    data=json.loads(migration.read_text()); step=data['step']; checkpoint=Path(data['checkpoint'])
    newroot=Path(data['new_config']['output']); previous=None
    manifest=dict(parser='vLLM 0.32 HarmonyParser with installed oss-harmony 0.0.11',resume_step=step,jobs=[],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/role_probe_training/on_policy_harmony').glob('*.py')},
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),migration=str(migration),recovery=str(newroot/'recovery.json'))
    env=dict(os.environ,VARIANT_CONFIG=str(config),VARIANT_REPORT=str(root))
    for stop in ends:
        command=['sbatch','--parsable','--export=ALL',f'--nice={nice}',f'--job-name=role-harmony-{name}-to{stop}',f'--output={root}/logs/%x-%j.out']
        if name=='appworld2700':command+=['--time=24:00:00']
        if previous:command+=[f'--dependency=afterok:{previous}']
        command += [str(CODE/'train.sbatch'),'--resume',str(checkpoint),'--stop-after',str(stop)]
        if not previous:command+=['--migration',str(migration)]
        job=subprocess.check_output(command,text=True,env=env).strip().split(';')[0]
        manifest['jobs'].append(dict(job_id=job,from_step=step,to_step=stop,command=command))
        (root/'harmony_jobs.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,job,step,stop,flush=True)
        previous=job;step=stop;checkpoint=newroot/f'checkpoints/step{stop:05d}'
    evaluations=json.loads((root/'evaluation_jobs.json').read_text())
    (root/'evaluation_jobs_before_harmony.json').write_text(json.dumps(evaluations,indent=2)+'\n')
    oldexport=evaluations['jobs']['export']['job_id']
    assert call(['squeue','-h','-j',oldexport,'-o','%T'])=='PENDING'
    subprocess.run(['scancel',oldexport],check=True)
    command=list(evaluations['jobs']['export']['command'])
    command=[f'--dependency=afterok:{previous}' if x.startswith('--dependency=') else '--nice=3000' if x.startswith('--nice=') else x for x in command]
    index=next(i for i,x in enumerate(command) if x.endswith('/post_training/export.sbatch'))
    command[index+1]=str(checkpoint)
    export=call(command).split(';')[0]
    evaluations['jobs']['export'].update(job_id=export,command=command)
    evaluations.update(training_job=previous,checkpoint=str(checkpoint),parser_training_manifest='harmony_jobs.json')
    # Each immediate export consumer gets a new dependency; its downstream DAG remains intact.
    for stage,item in evaluations['jobs'].items():
        if stage=='export':continue
        command=item['command']
        dependency=next(x for x in command if x.startswith('--dependency='))
        old=dependency.removeprefix('--dependency=')
        new=':'.join(export if token==oldexport else token for token in old.split(':'))
        args=['scontrol','update',f'JobId={item["job_id"]}','Nice=3000']
        if new!=old:
            args += [f'Dependency={new}']
            item['command']=[f'--dependency={new}' if x==dependency else x for x in command]
        subprocess.run(args,check=True)
        item['command']=['--nice=3000' if x.startswith('--nice=') else x for x in item['command']]
    (root/'evaluation_jobs.json').write_text(json.dumps(evaluations,indent=2)+'\n')
    print(name,'export reconnected',export,'final training',previous,flush=True)
