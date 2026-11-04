"""Check positive and negative controls through the pinned SWE sandbox and grader."""
import argparse
import json
from pathlib import Path
from inspect_ai import eval
from inspect_ai.solver import solver
from inspect_ai.util import sandbox
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.solvers import swe_bench_oracle_solver
from role_probe_training.trajectories.swe_snapshot import sandbox_config

@solver
def unchanged():
    async def solve(state, generate):
        await sandbox().write_file("/testbed/swe_control_marker.txt", "Nonfunctional control marker; no source edits.\n")
        return state
    return solve

p = argparse.ArgumentParser(__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
results = []
for name, agent, expected in [('gold', swe_bench_oracle_solver(), 1), ('unchanged', unchanged(), 0)]:
    logs = eval(swe_bench(sandbox_type='apptainer_repository_snapshot', sandbox_config=sandbox_config),
                model='mockllm/model', solver=agent,
                sample_id=['django__django-14155', 'sympy__sympy-13878', 'matplotlib__matplotlib-20859'],
                max_samples=2, log_dir=str(a.output/name), display='plain', fail_on_error=False)
    for log in logs:
        for sample in log.samples or []:
            result = {'control': name, 'sample_id': sample.id,
                      'scores': {k:v.value for k,v in (sample.scores or {}).items()},
                      'error': sample.error.message if sample.error else None,
                      'metadata': {k:v.metadata for k,v in (sample.scores or {}).items()}}
            results.append(result)
            (a.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
            print(name, sample.id, result['scores'], result['error'], flush=True)
assert len(results) == 6
assert all(r["error"] is None and list(r["scores"].values()) == [1 if r["control"] == "gold" else 0] for r in results), "Stock Inspect control failed"
print('Diagnostic complete; inspect gold verdicts and test logs before accepting the grader')
