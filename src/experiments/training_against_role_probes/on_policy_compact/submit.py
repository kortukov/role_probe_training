"""Plan compact evaluations; submit only with --submit. Does not replace jobs."""
import argparse
import json
import subprocess
from pathlib import Path

CODE = Path('src/experiments/training_against_role_probes')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--training-job', required=True, type=int)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    manifest = args.root / 'compact_evaluation_jobs.json'
    if args.submit:
        # Never silently duplicate the existing queued evaluations.
        if manifest.exists() or (args.root / 'evaluation_jobs.json').exists():
            raise FileExistsError('Evaluation manifest exists; reconcile the old jobs before submission')
        import os
        for key in ('AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT'):
            if not os.environ.get(key):
                raise ValueError(f'Missing inherited credential: {key}')
        (args.root / 'logs').mkdir(parents=True, exist_ok=True)
    record = {'training_job': str(args.training_job), 'checkpoint': str(args.checkpoint), 'jobs': {}}

    def stage(name, script, arguments, dependencies, options=()):
        command = ['sbatch', '--parsable', '--export=ALL',
                   '--exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02',
                   f'--output={args.root}/logs/%x-%A_%a.out',
                   '--dependency=afterok:' + ':'.join(dependencies),
                   *options, str(CODE / script), *map(str, arguments)]
        job = (subprocess.check_output(command, text=True).strip().split(';')[0]
               if args.submit else name)
        record['jobs'][name] = {'job_id': job, 'command': command}
        if args.submit:
            manifest.write_text(json.dumps(record, indent=2) + '\n')
        return job

    wrapper = 'on_policy_compact/evaluate.sbatch'
    prepare = stage('prepare', wrapper, ['prepare', args.root, args.checkpoint],
                    [str(args.training_job)], ['--time=08:00:00'])
    stage('appworld_retail', wrapper, ['app-retail', args.root, args.root.name],
          [prepare], ['--time=60:00:00'])
    swe = stage('swe_collection', 'swe_generous/collect.sbatch', [args.root],
                [prepare], ['--array=0-3%2', '--time=72:00:00'])
    stage('swe_finish', wrapper, ['swe-finish', args.root, swe],
          [swe], ['--time=30:00:00'])
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
