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
    'network': 'cat /etc/hosts /etc/nsswitch.conf /etc/resolv.conf; cat /proc/net/dev /proc/net/route; ip addr 2>/dev/null || true',
    'resolution': "python -c 'import socket; print(socket.getaddrinfo(\"localhost\", 0)); s=socket.socket(); s.bind((\"127.0.0.1\", 0)); print(s.getsockname())'",
    'loopback': "python -c 'import socket; s=socket.socket(); s.bind((\"127.0.0.1\",0)); s.listen(1); c=socket.socket(); c.settimeout(2); c.connect(s.getsockname()); print(\"loopback connected\")'",
    'requests_fixtures': "grep -nE 'httpbin|HTTPBIN|https?://|def httpbin|pytest_httpbin' conftest.py test_requests.py setup.cfg 2>/dev/null | head -100; cat conftest.py 2>/dev/null | head -160",
    'packages': "source /opt/miniconda3/bin/activate testbed; python -m pip show pytest-httpbin httpbin pytest-rerunfailures; cat /etc/hosts",
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
logs=eval(t,model='mockllm/model',solver=diagnose(),sample_id=['matplotlib__matplotlib-20859','psf__requests-1724','psf__requests-2931'],max_samples=2,log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[{'id':s.id,'error':s.error.message if s.error else None,'diagnostic':s.store} for log in logs for s in log.samples or []]
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
