# Running this experiment

- Code: [hx2/appworld.sbatch](../../../../../src/experiments/eval_role_probes/agentic_eval/appworld/hx2/appworld.sbatch) wrapping the
  unchanged [appworld_run.py](../../../../../src/experiments/eval_role_probes/agentic_eval/appworld/appworld_run.py), worker and
  [adapter](../../../../../src/role_probe_training/trajectories/appworld.py).
- Data: `runs/<job>/test_normal/trajectories.jsonl` (ignored; benchmark content stays off Git).
  `metrics/` keeps the summary, timing and manifest with task IDs removed.
- Jobs: pilot 1584579, normal test 1584591 (`logs/job_ids.txt`).
- Environments: this recorded cohort used the former `agentic_eval/swe_bench/env` (Inspect client) and
  `agentic_eval/server_current` (vLLM 0.32.0) projects. They are now the root environment
  (`uv sync --locked --extra vllm`, vLLM 0.27.1); `agentic_eval/appworld/env` (data download) is unchanged.
  See [ENVIRONMENTS.md](../../../../../agent_docs/ENVIRONMENTS.md) and [HX2.md](../../../../../agent_docs/HX2.md).

```bash
export APPTAINER_CACHEDIR=/hx2-weka/home/xy2222/.cache/apptainer
apptainer pull .cache/appworld/appworld.sif docker://ghcr.io/stonybrooknlp/appworld:latest
uv run --project src/experiments/eval_role_probes/agentic_eval/appworld/env --frozen --no-sync appworld install
uv run --project src/experiments/eval_role_probes/agentic_eval/appworld/env --frozen --no-sync appworld download data --root "$PWD/.cache/appworld"
sbatch src/experiments/eval_role_probes/agentic_eval/appworld/hx2/appworld.sbatch --pilot
sbatch --job-name=appworld-hx2-test_normal src/experiments/eval_role_probes/agentic_eval/appworld/hx2/appworld.sbatch test_normal
```

Probe scoring of these histories: [whole_score.sbatch](../../../../../src/experiments/eval_role_probes/agentic_eval/probe_scoring/hx2/whole_score.sbatch)
with the trajectories file, an output root and the probe arm directories.
