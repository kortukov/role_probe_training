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
 'astropy_parameters': "grep -n -B 15 -A 20 'def test_compose_roundtrip' astropy/units/tests/test_units.py; grep -n -A 70 'def _idval' /opt/miniconda3/envs/testbed/lib/python3.6/site-packages/_pytest/python.py",
 'django_templates': "git log -1 --format='%H %s'; git status --short | head -25; git ls-tree -r HEAD --name-only | grep 'password_change_done.html'; ls django/contrib/admin/templates/registration; grep -n -A 60 'class AuthTemplateTests' tests/auth_tests/test_templates.py; sed -n '1,130p' tests/auth_tests/templates.py",
 'django_runtime': "source /opt/miniconda3/bin/activate testbed; python -c 'import django,sys; print(django.__file__); print(django.VERSION); print(sys.path)'; grep -n 'TEMPLATE\|INSTALLED_APPS\|TEMPLATES' tests/auth_tests/test_templates.py tests/test_sqlite.py tests/runtests.py",
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
logs=eval(t,model='mockllm/model',solver=diagnose(),sample_id=['astropy__astropy-7606','django__django-10097'],max_samples=2,log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[{'id':s.id,'error':s.error.message if s.error else None,'diagnostic':s.store} for log in logs for s in log.samples or []]
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
