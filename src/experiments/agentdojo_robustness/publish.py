"""Publish only this experiment on a dedicated branch, using a temporary index."""
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path('experiment_reports/prompt_injection/agentdojo_same_batch_27000')
BRANCH = 'agentdojo-same-batch-27000'

def git(*args, env=None):
    return subprocess.check_output(['git', *args], env=env, text=True).strip()

with tempfile.TemporaryDirectory() as d:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(d)/'index'))
    ref = git('ls-remote', 'origin', f'refs/heads/{BRANCH}')
    if ref:
        git('fetch', 'origin', BRANCH)
        parent = git('rev-parse', 'FETCH_HEAD')
    else:
        parent = git('rev-parse', 'HEAD')
    git('read-tree', parent, env=env)
    git('add', '--', str(ROOT), 'src/experiments/agentdojo_robustness', env=env)
    tree = git('write-tree', env=env)
    commit = git('commit-tree', tree, '-p', parent, '-m', 'Save AgentDojo clean and attacked evaluation results', env=env)
    git('push', 'origin', f'{commit}:refs/heads/{BRANCH}')
    print(f'https://github.com/kortukov/role_probe_training/tree/{BRANCH}/{ROOT}')
