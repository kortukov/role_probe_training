"""Legacy evaluation layout; use on_policy_compact/submit.py for new runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

CODE = Path('src/experiments/training_against_role_probes')
BASE = Path('experiment_reports/training_against_role_probes/on_policy')
RUNS = [('appworld270_batch64', '1787803', 'epoch1_repaired', 5),
        ('appworld270_batch16', '1787804', 'epoch1_repaired', 17),
        ('appworld2700', '1798639', 'colocated_epoch1', 169)]
EXCLUDE = 'vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.submit:
        for key in ('AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT'):
            if not os.environ.get(key):
                raise ValueError(f'Missing inherited retail credential: {key}')
        for name, *_ in RUNS:
            if (BASE/name/'evaluation_jobs.json').exists():
                raise FileExistsError(f'{name}: evaluation already attached; inspect manifest')
    for name, training, run, step in RUNS:
        root = BASE/name
        checkpoint = root/f'runs/{run}/checkpoints/step{step:05d}'
        record = dict(training_job=training, checkpoint=str(checkpoint), jobs={},
                      protocol=dict(appworld_tasks=168, retail_tasks=114, swe_tasks=500,
                                    swe='swe_inspect_generous_v1', probe='fresh layer 12; 250 passages x 5 roles',
                                    scoring='whole histories; explicit native-context exclusions'))
        if args.submit:
            (root/'logs').mkdir(exist_ok=True)
            (root/'post_training').mkdir(exist_ok=True)

        def submit(stage, script, arguments, dependencies, options=()):
            path=CODE/script
            assert path.is_file(), path
            command=['sbatch','--parsable','--export=ALL',
                     f'--exclude={EXCLUDE}', f'--output={root}/logs/%x-%A_%a.out',
                     '--dependency=afterok:'+':'.join(dependencies), *options,
                     str(path), *map(str,arguments)]
            job = (subprocess.check_output(command,text=True).strip().split(';')[0]
                   if args.submit else stage)
            record['jobs'][stage]=dict(job_id=job,command=command,
                script_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if args.submit:
                (root/'evaluation_jobs.json').write_text(json.dumps(record,indent=2)+'\n')
            print(name,stage,job,flush=True)
            return job

        export=submit('export','post_training/export.sbatch',
                      [checkpoint,root/'data/post_training/model'],[training])
        fit=submit('probe_fit','post_training/probes.sbatch',['fit','--root',root],[export])
        collect=submit('appworld_retail_collection','kl_iteration/collect.sbatch',
                       [root,name],[export],['--array=0-2%2'])
        assemble=submit('appworld_retail_assembly','kl_iteration/assemble_benchmark.sbatch',
                        [root,collect,'--benchmarks','appworld_test','tau2_retail'],[collect])
        swe=submit('swe_collection','swe_generous/collect.sbatch',[root],[export],
                   ['--array=0-3%2','--time=3-00:00:00'])
        swe_assemble=submit('swe_assembly','swe_generous/assemble.sbatch',[root,swe],[swe])
        for index,benchmark in enumerate(('appworld_test','tau2_retail','swe_bench')):
            if benchmark=='swe_bench':
                score=submit(benchmark+'_probes','swe_generous/score.sbatch',
                             [root],[fit,swe_assemble],['--time=24:00:00'])
            else:
                score=submit(benchmark+'_probes','post_training/whole_score.sbatch',
                             [root,'--overlength','exclude'],[fit,assemble],
                             [f'--array={index}','--time=12:00:00'])
            submit(benchmark+'_results','on_policy_evaluation/results.sbatch',
                   [root,benchmark],[score])


if __name__=='__main__':
    main()
