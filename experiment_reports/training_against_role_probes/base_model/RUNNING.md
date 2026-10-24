# Reproduction

Sources and actual job IDs are recorded in [post_training/jobs.json](post_training/jobs.json).
The `data/post_training/*/trajectories.jsonl` files are links to the original
canonical agentic datasets. Task-score provenance is in
[post_training/task_results.json](post_training/task_results.json).

Use the shared training runtime and these entry points from the repository root:

```bash
sbatch --array=0,2 src/experiments/training_against_role_probes/post_training/whole_score.sbatch experiment_reports/training_against_role_probes/base_model --base
sbatch --array=1 src/experiments/training_against_role_probes/post_training/whole_score.sbatch experiment_reports/training_against_role_probes/base_model --base --overlength exclude --output experiment_reports/training_against_role_probes/base_model/predictions/post_training_whole/tau2_retail
```

For a fresh rerun, submit the first command with `--array=0,2`; retail uses the
second command so the one over-limit history is explicitly recorded. The unchanged
model is dequantized to BF16 and uses flex attention for full contexts. Pre-MLP
hooks collect all five layers together. No chunks, truncation, or learned adapter
updates are applied. Per-token NPZ predictions and large input data remain local.

After all cohorts finish, run `post_training/compare.py` and
`post_training/refresh_base_reports.py` under `src/experiments/training_against_role_probes/`
with `uv run --frozen --no-sync python`. The parent
[RUNNING.md](../RUNNING.md#fresh-rollout-evaluation) describes shared conventions.
