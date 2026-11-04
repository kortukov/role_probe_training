"""Publish only the requested experiment's small validated artifacts to GitHub."""
import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    relative=args.root
    if relative not in (Path('experiment_reports/training_against_role_probes/on_policy_epochs'),
                         Path('experiment_reports/training_against_role_probes/off_policy_epochs')):
        raise ValueError('Unexpected publication scope')
    root=relative.resolve()
    git_root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    # Serialize publication from these two experiments without touching the user's index.
    with (root.parent/'.epoch-publication.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        def git(*arguments, cwd=git_root):
            return subprocess.check_output(['git',*map(str,arguments)],cwd=cwd,text=True).strip()
        git('fetch','origin')
        with tempfile.TemporaryDirectory(prefix='role-epoch-publish-') as temporary:
            work=Path(temporary)/'checkout'
            git('worktree','add','--detach',work,'origin/main')
            try:
                destination=work/relative
                destination.mkdir(parents=True,exist_ok=True)
                paths=[*root.glob('*.md'),*root.glob('*.json'),root/'.gitignore']
                paths += list((root/'checkpoints').glob('epoch*/post_training/results/*.json'))
                paths += list((root/'checkpoints').glob('epoch*/post_training/probe_fit.json'))
                paths += list((root/'checkpoints').glob('epoch*/post_training/probes/*'))
                for source in paths:
                    if source.is_file():
                        target=destination/source.relative_to(root)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(source,target)
                git('add','--',relative,cwd=work)
                if not git('diff','--cached','--name-only',cwd=work):
                    return
                git('commit','-m',f'Update {root.name} retail checkpoint results',cwd=work)
                for attempt in range(5):
                    push=subprocess.run(['git','push','origin','HEAD:main'],cwd=work)
                    if push.returncode==0:
                        return
                    git('fetch','origin',cwd=work)
                    git('rebase','origin/main',cwd=work)
                raise RuntimeError('GitHub push failed after five attempts')
            finally:
                git('worktree','remove','--force',work)


if __name__=='__main__':
    main()
