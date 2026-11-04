"""Publish this experiment to main without touching the shared working index."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT='experiment_reports/prompt_injection/agentdojo_same_batch_27000'
PATHS=[ROOT,'src/experiments/agentdojo_roles','src/role_probe_training/post_training/agentdojo.py']

def git(*args,env=None):
    return subprocess.check_output(['git',*args],text=True,env=env).strip()

for attempt in range(3):
    with tempfile.TemporaryDirectory() as d:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(d)/'index'))
        git('fetch','origin','main')
        parent=git('rev-parse','FETCH_HEAD')
        git('read-tree',parent,env=env)
        git('add','--',*PATHS,env=env)
        tree=git('write-tree',env=env)
        if tree==git('rev-parse',parent+'^{tree}'):
            print('Main already contains current AgentDojo role artifacts');break
        commit=git('commit-tree',tree,'-p',parent,'-m','Report fresh-probe AgentDojo role perception on clean and attacked histories',env=env)
        result=subprocess.run(['git','push','origin',commit+':refs/heads/main'])
        if result.returncode==0:
            print('https://github.com/kortukov/role_probe_training/tree/main/'+ROOT);break
else:raise RuntimeError('Publication failed; local artifacts preserved')
