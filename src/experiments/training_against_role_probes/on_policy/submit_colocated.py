"""Queue diagnostics, batch-16/restart pilots, then a fail-closed full run."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('experiment_reports/training_against_role_probes/on_policy/appworld2700')
CODE=Path('src/experiments/training_against_role_probes/on_policy')

def main():
    manifest=ROOT/'colocated_jobs.json'
    if manifest.exists():raise FileExistsError(manifest)
    jobs={}
    def submit(name,options,script,arguments=()):
        command=['sbatch','--parsable',*options,str(CODE/script),*arguments]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        jobs[name]=dict(job_id=job,command=command)
        manifest.write_text(json.dumps(jobs,indent=2)+'\n')
        print(name,job,flush=True)
        return job
    check=submit('colocation_check',[],'colocation_check.sbatch')
    check_dir=str(ROOT/'runs/colocation'/check)
    exports=f'--export=ALL,COLOCATION_CHECK_DIR={check_dir}'
    pilot=submit('pilot',[f'--dependency=afterok:{check}','--time=06:00:00',exports],
                 'colocated_train.sbatch',['--stop-after','1'])
    run=ROOT/'runs/colocated_epoch1'
    resumed=submit('resume_pilot',[f'--dependency=afterok:{pilot}','--time=06:00:00',exports],
        'colocated_train.sbatch',['--stop-after','2','--resume',str(run/'checkpoints/step00001')])
    gate=submit('gate',[f'--dependency=afterok:{resumed}'],'colocated_gate.sbatch')
    submit('training',[f'--dependency=afterok:{gate}',exports],'colocated_train.sbatch',
           ['--resume',str(run/'checkpoints/step00002')])
    paths=list(Path('src/role_probe_training/on_policy').glob('*.py'))+list(CODE.glob('*.py'))
    paths+=list(CODE.glob('*.sbatch'))+list(CODE.glob('*.json'))
    jobs['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest.write_text(json.dumps(jobs,indent=2)+'\n')

if __name__=='__main__':main()
