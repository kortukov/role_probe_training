"""Submit the fixed evaluation DAG for an already submitted training run."""
import argparse
import json
from pathlib import Path
import subprocess

BASE = Path('experiment_reports/training_against_role_probes')
CODE = Path('src/experiments/training_against_role_probes')
EXCLUDE = 'vca-gpu-0701-03,vca-gpu-0701-05'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('variant', choices=['agentic800','probe_data'])
    parser.add_argument('--training', required=True)
    parser.add_argument('--native-gate', required=True)
    args = parser.parse_args()
    root = BASE/'kl_iteration'/args.variant
    path = root/'jobs.json'
    if path.exists(): raise FileExistsError(path)
    jobs = dict(training=args.training, native_gate=args.native_gate)
    def submit(stage, script, arguments=(), dependencies=(), options=()):
        command = ['sbatch','--parsable',f'--exclude={EXCLUDE}',*options]
        if dependencies: command += ['--dependency=afterok:'+':'.join(map(str,dependencies))]
        command += [str(script), *map(str,arguments)]
        job = subprocess.check_output(command,text=True).strip().split(';')[0]
        jobs[stage] = job
        path.write_text(json.dumps(jobs,indent=2)+'\n')
        print(stage, job, flush=True)
        return job
    export = submit('export', CODE/'post_training/export.sbatch',
        [root/'runs/training/epoch1/checkpoints/step00169',root/'data/post_training/model'],[args.training])
    fit = submit('probe_fit',CODE/'post_training/probes.sbatch',['fit','--root',root],[export])
    collect = submit('collect',CODE/'post_training/collect.sbatch',[root,'kl-'+args.variant],
        [export],['--array=0-2%2'])
    swe = submit('swe_collect',CODE/'post_training/swe_native.sbatch',[root,'full'],
        [export,args.native_gate],['--array=0-3%2'])
    assemble = submit('assemble',CODE/'kl_iteration/assemble.sbatch',[root,collect,swe],[collect,swe])
    score = submit('score',CODE/'post_training/whole_score.sbatch',[root,'--overlength','exclude'],
        [assemble,fit],['--array=0-2%2','--time=08:00:00'])
    submit('report',CODE/'kl_iteration/finalize.sbatch',[root],[score])


if __name__ == '__main__':
    main()
