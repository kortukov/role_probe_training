"""Queue AppWorld and retail only after expanded-data training succeeds."""
import argparse
import json
from pathlib import Path
import subprocess

CODE = Path('src/experiments/training_against_role_probes')
ROOT = Path('experiment_reports/training_against_role_probes/data_iteration/same_batch_27000')
EXCLUDE = 'vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training', required=True)
    args = parser.parse_args()
    path = ROOT / 'jobs.json'
    jobs = json.loads(path.read_text())
    if 'export' in jobs:
        raise ValueError('Evaluation already submitted; inspect recorded jobs before resubmitting')

    def submit(stage, script, arguments, dependencies, options=()):
        command = ['sbatch', '--parsable', f'--exclude={EXCLUDE}', *options,
                   '--dependency=afterok:' + ':'.join(map(str, dependencies)),
                   str(CODE / script), *map(str, arguments)]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        jobs[stage] = job
        path.write_text(json.dumps(jobs, indent=2) + '\n')
        print(stage, job, flush=True)
        return job

    export = submit('export', 'post_training/export.sbatch',
                    [ROOT / 'runs/training/epoch1/checkpoints/step01688',
                     ROOT / 'data/post_training/model'], [args.training])
    fit = submit('probe_fit', 'post_training/probes.sbatch', ['fit', '--root', ROOT], [export])
    collect = submit('evaluation_collection', 'kl_iteration/collect.sbatch',
                     [ROOT, 'same-batch-27000'], [export], ['--array=0-2%2'])
    assemble = submit('evaluation_assembly', 'kl_iteration/assemble_benchmark.sbatch',
                      [ROOT, collect, '--benchmarks', 'appworld_test', 'tau2_retail'], [collect])
    submit('role_scoring', 'post_training/whole_score.sbatch',
           [ROOT, '--overlength', 'exclude'], [fit, assemble],
           ['--array=0-1%2', '--time=08:00:00'])


if __name__ == '__main__':
    main()
