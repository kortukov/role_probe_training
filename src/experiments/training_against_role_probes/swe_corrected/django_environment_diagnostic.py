"""Inspect isolated container networking and test fixture assumptions; no model calls."""
import argparse
import json
from pathlib import Path
from inspect_ai import eval
from inspect_ai.solver import solver
from inspect_ai.util import sandbox
from inspect_evals.swe_bench import swe_bench
from role_probe_training.trajectories.apptainer_mirror import sandbox_config

COMMANDS = {
 'settings': "sed -n '1,22p' tests/auth_tests/test_templates.py; sed -n '55,100p' tests/runtests.py; sed -n '135,225p' tests/runtests.py; cat tests/test_sqlite.py",
 'sqlite': "source /opt/miniconda3/bin/activate testbed; python -c 'import sqlite3; print(sqlite3.sqlite_version); print(sqlite3.__file__)'; ls -l /opt/miniconda3/envs/testbed/lib/libsqlite*",
 'template_paths': "ls -ld django/contrib/admin/templates django/contrib/admin/templates/registration; head -12 django/contrib/admin/templates/registration/password_change_done.html; git status --short; sed -n '1,100p' tests/generic_inline_admin/tests.py",
}


@solver
def diagnose():
    async def solve(state, generate):
        outputs = {}
        for name, cmd in COMMANDS.items():
            result = await sandbox().exec(['bash', '-c', cmd])
            outputs[name] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        state.store.set('network_diagnostic', outputs)
        return state
    return solve

p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
t=swe_bench(sandbox_type='apptainer',sandbox_config=sandbox_config);t.scorer=[]
logs=eval(t,model='mockllm/model',solver=diagnose(),sample_id=['django__django-10097'],max_samples=2,log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[{'id':s.id,'error':s.error.message if s.error else None,'diagnostic':s.store} for log in logs for s in log.samples or []]
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
