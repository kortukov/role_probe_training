"""Compare Python environments reached by direct sandbox, python, and bash_session."""
import argparse
import json
import shlex
from pathlib import Path
from inspect_ai import eval
from inspect_ai.solver import solver
from inspect_ai.tool import python, bash_session
from inspect_ai.util import sandbox
from inspect_evals.swe_bench import swe_bench
from role_probe_training.trajectories.apptainer_mirror import sandbox_config

CODE="import os,sys,importlib.util,json; print(json.dumps({'executable':sys.executable,'path':sys.path,'env':{k:os.environ.get(k) for k in ['PATH','PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_DEFAULT_ENV']},'modules':{m:bool(importlib.util.find_spec(m)) for m in ['pytest','mpmath','asgiref','numpy']}}))"

@solver
def check_tools():
    async def solve(state,generate):
        box=sandbox();results={}
        r=await box.exec(['python','-c',CODE]);results['direct']={'stdout':r.stdout,'stderr':r.stderr}
        results['python_tool']=await python()(code=CODE)
        results['bash_session']=await bash_session()(action='type_submit',input='python -c '+shlex.quote(CODE))
        results['bash_session_activated']=await bash_session()(action='type_submit',input='source /opt/miniconda3/bin/activate testbed; python -c '+shlex.quote(CODE))
        r=await box.exec(['bash','-c','source /opt/miniconda3/bin/activate testbed; python -c "$1"','bash',CODE]);results['direct_activated']={'stdout':r.stdout,'stderr':r.stderr}
        state.store.set('environment_probe',results)
        return state
    return solve

p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--activate-testbed',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
if a.activate_testbed:
    from role_probe_training.trajectories.swe_testbed import sandbox_config
t=swe_bench(sandbox_type='apptainer',sandbox_config=sandbox_config);t.scorer=[]
logs=eval(t,solver=check_tools(),model='mockllm/model',sample_id=['django__django-14155','sympy__sympy-13878'],max_samples=2,log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[{'id':s.id,'error':s.error.message if s.error else None,'environment':s.store} for log in logs for s in log.samples or []]
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
