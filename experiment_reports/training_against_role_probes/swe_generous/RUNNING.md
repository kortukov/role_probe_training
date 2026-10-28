# Reproduction

The active configuration and job dependencies are recorded in [pipeline.json](pipeline.json).
The cohort contains the base model and all six completed trained models, including
same-batch KL trained on 27,000 AppWorld trajectories. The on-policy experiment
has no completed production model and is not included.

## Shared evaluation protocol

Every model receives all 500 SWE-bench Verified test tasks, one attempt per task,
with seed 123, high reasoning effort, temperature 1, BF16 weights, a native
131,072-token context, and at most 16,384 output tokens per generation. The
message budget is 2,000, the per-task safety timeout is 7,200 seconds, and there
is no cumulative token budget. The stock tool timeout is 210 seconds.

Pinned versions are Inspect Evals 0.25.0, Inspect AI 0.3.283 and SWE-bench 4.1.0.
Dataset revision: `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
The agent and scorer are stock Inspect. The sandbox activates repository Python,
provides localhost resolution and snapshots initial files. It retains isolated
networking and does not add task-specific fixtures or scoring repairs.

This restores the earlier message budget. The historical 247/500 baseline was
saved-patch diagnostic regrading under multiple repair protocols; it is not the
baseline for this cohort. The superseded 30-message run and all original attempts
remain in their separate directories.

## Execution and artifacts

Scripts are under `src/experiments/training_against_role_probes/swe_generous/`.
For each model root, collection uses four shards with at most two running at once:

```bash
sbatch --array=0-3%2 src/experiments/training_against_role_probes/swe_generous/collect.sbatch MODEL_ROOT
sbatch --dependency=afterok:COLLECTION src/experiments/training_against_role_probes/swe_generous/assemble.sbatch MODEL_ROOT COLLECTION
sbatch --dependency=afterok:ASSEMBLY src/experiments/training_against_role_probes/swe_generous/score.sbatch MODEL_ROOT
```

These are reproduction commands; do not resubmit existing cohorts. Submitted
trained collections depend on baseline assembly. Assembly validates 500 unique
assigned IDs, settings and exports. Scoring replays each saved history whole,
using the independently fitted model-specific layer-12 probe. Over-context
histories remain in task denominators and are excluded only from role metrics.

Large artifacts stay on cluster storage beneath each model root:
`runs/swe_inspect_generous/COLLECTION/shardN/swe_bench/`,
`data/swe_inspect_generous/evaluation/trajectories.jsonl`, and
`predictions/swe_inspect_generous/`. Small task summaries are in
`post_training/swe_inspect_generous_task_results.json`. Source hashes and model
revisions are recorded in per-shard manifests. All task results are required for
the task report; all seven probe results are required for the final report.
