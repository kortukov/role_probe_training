"""Plan or submit fixed-data continuation from a frozen source snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

CODE = Path('src/experiments/training_against_role_probes/same_batch_epochs')
ROOT = Path('experiment_reports/training_against_role_probes/same_batch_epochs')
ORIGINAL = Path('experiment_reports/training_against_role_probes/kl_iteration/same_batch')
CHECKPOINT = ORIGINAL/'runs/training/epoch1/checkpoints/step00169'
EXCLUDE = 'vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def baseline():
    from report import validate
    tasks = json.loads((ORIGINAL/'post_training/task_results.json').read_text())
    export = json.loads((ORIGINAL/'data/post_training/model/export.json').read_text())
    if digest(CHECKPOINT/'adapter/adapter_model.safetensors') != export['adapter_sha256']:
        raise ValueError('Original evaluated model is not the continuation checkpoint')
    for benchmark, expected in (('appworld_test',168), ('tau2_retail',114)):
        source = ORIGINAL/f'predictions/post_training/{benchmark}/summary.json'
        probe = json.loads(source.read_text())
        if probe['model_revision'] != export['adapter_sha256']:
            raise ValueError('Original probe scored a different model')
        if probe['probe_sha256'] != digest(ORIGINAL/'post_training/probes/layer_12.npz'):
            raise ValueError('Original evaluation probe hash differs')
        result = dict(benchmark=benchmark,task=tasks[benchmark],probe=probe,
                      checkpoint=str(CHECKPOINT),model_revision=export['adapter_sha256'],
                      source_sha256={str(p):digest(p) for p in (source,ORIGINAL/'post_training/task_results.json',
                                                               ORIGINAL/'data/post_training/model/export.json')},
                      reused_saved_evaluation=True)
        validate(result, expected)
        target = ROOT/f'evaluations/epoch001/post_training/results/{benchmark}.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(result,indent=2)+'\n')
    for filename in ('layer_12.npz','layer_12.json'):
        target = ROOT/'evaluations/epoch001/post_training/probes'/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ORIGINAL/'post_training/probes'/filename,target)


def freeze(workspace):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot = workspace.parent/'role_probe_training_runs'/f'same_batch_epochs_{stamp}'
    snapshot.mkdir(parents=True)
    shutil.copytree(workspace/'src',snapshot/'src',ignore=shutil.ignore_patterns('.venv','.uv-cache','__pycache__','*.pyc'))
    for parent, directories, _ in os.walk(workspace/'src'):
        for name in list(directories):
            if name in ('.venv','.uv-cache','__pycache__'):
                directories.remove(name)
                if name == '.venv':
                    source=Path(parent)/name
                    (snapshot/source.relative_to(workspace)).symlink_to(source.resolve(),target_is_directory=True)
    for name in ('pyproject.toml','uv.lock','.python-version','README.md'):
        shutil.copy2(workspace/name,snapshot/name)
    for name in ('.git','.venv','.tools','.cache','experiment_reports','submodules'):
        (snapshot/name).symlink_to(workspace/name,target_is_directory=True)
    hashes = {str(p.relative_to(snapshot)):digest(p) for p in (snapshot/'src').rglob('*')
              if p.is_file() and '.venv' not in p.parts and p.suffix in ('.py','.sbatch','.json','.toml','.lock','.sh')}
    record = dict(workspace=str(workspace),snapshot=str(snapshot),created_utc=stamp,source_sha256=hashes,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  submodules=subprocess.check_output(['git','submodule','status'],text=True).strip())
    (snapshot/'FROZEN_SOURCE.json').write_text(json.dumps(record,indent=2)+'\n')
    return snapshot, record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--baseline-only',action='store_true')
    args = parser.parse_args()
    if args.baseline_only:
        baseline()
        return
    workspace = Path.cwd().resolve()
    c = json.loads((CODE/'config.json').read_text())
    original = json.loads((ORIGINAL/'runs/training/epoch1/manifest.json').read_text())
    for key in ('model','probe','trajectories','expected_trajectories','validation','probe_documents','kl_anchors',
                'seed','learning_rate','kl_coefficient','refit_steps','kl_sampling'):
        if c[key] != original[key]:
            raise ValueError(f'Continuation changes optimization or input setting: {key}')
    if not (CHECKPOINT/'training.pt').is_file():
        raise FileNotFoundError('Original optimizer/probe recovery state missing')
    for key, expected in original['sha256'].items():
        if digest(c[key]) != expected:
            raise ValueError(f'Original training input changed: {key}')
    for name, expected in original['source_sha256'].items():
        source = workspace/'src/role_probe_training/training'/Path(name).name
        if digest(source) != expected:
            raise ValueError(f'Original shared training implementation changed: {source}')
    snapshot, provenance = workspace, None
    if args.submit:
        if (ROOT/'jobs.json').exists():
            raise FileExistsError('Existing jobs manifest: reconcile it before any recovery')
        for key in ('AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT'):
            if not os.environ.get(key):
                raise ValueError(f'Missing inherited retail credential: {key}')
        (ROOT/'logs').mkdir(parents=True,exist_ok=True)
        baseline()
        snapshot, provenance = freeze(workspace)
    record = dict(source_snapshot=str(snapshot),frozen_source=provenance,configuration=c,
                  original_checkpoint=str(CHECKPOINT),original_input_sha256=original['sha256'],
                  starting_step=169,total_updates=845,additional_updates=676,evaluated_epochs=[1,2,3,5],
                  original_checkpoint_sha256={name:digest(CHECKPOINT/name) for name in
                                              ('training.pt','adapter/adapter_model.safetensors','layer_12.npz')},jobs={})
    environment = dict(os.environ,PYTHONPATH=str(snapshot/'src'),WANDB_MODE='offline')
    for key in list(environment):
        if key.startswith('SLURM_CPU_BIND'):
            environment.pop(key)

    def stage(name, script, arguments, dependency=None):
        command = ['sbatch','--parsable','--export=ALL',f'--chdir={snapshot}',f'--exclude={EXCLUDE}',
                   '--ntasks=1',f'--output={workspace/ROOT}/logs/%x-%j.out']
        if dependency:
            command += [f'--dependency=afterok:{dependency}']
        command += [str(script),*map(str,arguments)]
        job = (subprocess.check_output(command,text=True,cwd=snapshot,env=environment).strip().split(';')[0]
               if args.submit else name)
        record['jobs'][name] = dict(job_id=job,command=command,script_sha256=digest(script))
        if args.submit:
            target = ROOT/'jobs.json'
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record,indent=2)+'\n')
            temporary.replace(target)
        return job

    train = stage('training',CODE/'train.sbatch',[CODE/'config.json','--resume',CHECKPOINT])
    for epoch in (2,3,5):
        stage(f'evaluate_epoch{epoch:03d}',CODE/'evaluate.sbatch',
              [ROOT/f'evaluations/epoch{epoch:03d}',Path(c['output'])/f'epochs/epoch{epoch:03d}'],train)
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
