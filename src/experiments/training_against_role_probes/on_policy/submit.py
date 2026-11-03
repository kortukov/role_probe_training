"""Submit fresh pilot, restart pilot, validation gate and production continuation."""
import json
from pathlib import Path
import subprocess

ROOT = Path('experiment_reports/training_against_role_probes/on_policy/appworld2700')
CODE = Path('src/experiments/training_against_role_probes/on_policy')


def main():
    path = ROOT/'jobs.json'
    if path.exists():
        raise FileExistsError('Pipeline already submitted; inspect jobs.json instead of duplicating it')
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    jobs = {}

    def submit(name, script, arguments=(), dependency=None, options=()):
        command = ['sbatch', '--parsable', *options]
        if dependency:
            command += [f'--dependency=afterok:{dependency}']
        command += [str(CODE/script), *map(str, arguments)]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        jobs[name] = {'job_id': job, 'command': command}
        path.write_text(json.dumps(jobs, indent=2)+'\n')
        print(name, job, flush=True)
        return job

    pilot = submit('pilot', 'train.sbatch', ['--stop-after', 1], options=['--time=06:00:00'])
    resume = submit('resume_pilot', 'train.sbatch', ['--stop-after', 2, '--resume',
        ROOT/'runs/epoch1/checkpoints/step00001'], pilot, ['--time=06:00:00'])
    gate = submit('gate', 'gate.sbatch', dependency=resume)
    submit('training', 'train.sbatch', ['--resume', ROOT/'runs/epoch1/checkpoints/step00002'], gate)


if __name__ == '__main__':
    main()
