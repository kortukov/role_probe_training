"""Check positive and negative controls through the pinned SWE sandbox and grader."""
import argparse
import json
from pathlib import Path
from inspect_ai import eval
from inspect_ai.solver import solver
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.solvers import swe_bench_oracle_solver
from role_probe_training.trajectories.swe_snapshot import sandbox_config
from role_probe_training.trajectories.swe_scoring_snapshot import fresh_swe_scorer

@solver
def unchanged():
    async def solve(state, generate):
        return state
    return solve

p = argparse.ArgumentParser(__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
results = []
for name, agent, expected in [('gold', swe_bench_oracle_solver(), 1)]:
    logs = eval(swe_bench(sandbox_type='apptainer_repository_snapshot', sandbox_config=sandbox_config, scorer=fresh_swe_scorer()),
                model='mockllm/model', solver=agent,
                sample_id=['psf__requests-1142', 'psf__requests-6028'],
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
assert len(results) == 2
print('Diagnostic complete; inspect gold verdicts and test logs before accepting the grader')
