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
    'runtime': "source /opt/miniconda3/bin/activate testbed; python -V; python -m pip show pytest appdirs pylint astropy; python -c 'import sys; print(sys.path)'",
    'astropy_ids': "sed -n '/def test_compose_roundtrip/,+22p' astropy/units/tests/test_units.py; rg -n 'parametrize|compose_roundtrip' astropy/units/tests/test_units.py | tail -15",
    'requests_fixtures': "cat conftest.py 2>/dev/null; grep -nE 'HTTPBIN|httpbin|TARPIT|conflicting_post|mixed_case|timeout' test_requests.py tests/test_requests.py 2>/dev/null | head -100",
    'dependency_metadata': "grep -nE 'appdirs|install_requires|pytest|python_requires' setup.py setup.cfg pyproject.toml 2>/dev/null",
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
logs=eval(t,model='mockllm/model',solver=diagnose(),sample_id=['astropy__astropy-7606','pylint-dev__pylint-4661','psf__requests-1766','psf__requests-5414'],max_samples=2,log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[{'id':s.id,'error':s.error.message if s.error else None,'diagnostic':s.store} for log in logs for s in log.samples or []]
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
