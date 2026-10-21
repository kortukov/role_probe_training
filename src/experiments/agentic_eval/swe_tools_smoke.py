from inspect_ai import eval
from inspect_ai.solver import solver
from inspect_ai.tool import bash_session, python, text_editor
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.solvers import swe_bench_oracle_solver
from role_probe_training.trajectories.apptainer import sandbox_config

@solver
def check_tools():
 async def solve(state,generate):
  result=await python()(code="print(12345)")
  assert '12345' in str(result),result
  result=await bash_session()(action='type_submit',input='printf shell-ok')
  assert 'shell-ok' in str(result),result
  result=await text_editor()(command='view',path='/testbed/setup.py')
  assert 'setup' in str(result),result
  return state
 return solve
logs=eval(swe_bench(sandbox_type='apptainer',sandbox_config=sandbox_config),model='mockllm/model',solver=[check_tools(),swe_bench_oracle_solver()],sample_id=['django__django-14155'],log_dir='experiment_reports/agentic/swe_bench_full/runs/tools_smoke',display='plain',fail_on_error=False)
for log in logs:
 for sample in log.samples or []:
  print(sample.id, 'scores', {k:v.value for k,v in sample.scores.items()}, 'error', sample.error.message if sample.error else None)
  assert not sample.error
