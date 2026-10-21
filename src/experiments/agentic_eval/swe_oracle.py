"""Validate the isolated SWE-bench environment and upstream test grading."""
from inspect_ai import eval
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.solvers import swe_bench_oracle_solver
from role_probe_training.trajectories.apptainer import sandbox_config

logs = eval(swe_bench(sandbox_type="apptainer", sandbox_config=sandbox_config),
            model="mockllm/model", solver=swe_bench_oracle_solver(),
            sample_id=["django__django-14155"], log_dir="experiment_reports/agentic/swe_bench_full/runs/oracle", display="plain", fail_on_error=False)
for log in logs:
    print(log.status, log.results)
    for sample in log.samples or []:
        print(sample.id, sample.scores, sample.error)
