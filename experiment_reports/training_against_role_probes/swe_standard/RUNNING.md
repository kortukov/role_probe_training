# Reproduction and execution

The authoritative job IDs and settings are in [pipeline.json](pipeline.json).
The previous model-card investigation has stopped at the user's request. Its
saved attempts remain under `swe_corrected*`; none enter this comparison.

This protocol uses unmodified inspect-evals 0.25.0, inspect-ai 0.3.283 and
swebench 4.1.0. Installed Python source hashes were checked against distribution
RECORD entries. Dependencies remain in the locked SWE and server_current
projects; no shared environment was changed.

Use the stock `inspect_evals.swe_bench` task, solver and scorer. The scorer runs
in the agent's sandbox, as upstream specifies. No fresh-image custom grader,
task-specific HTTP fixtures, test-name aliases, or altered test commands are used.
The cluster's Apptainer provider retains digest-pinned official images, activates
the repository interpreter, supplies localhost entries and snapshots initial
files before the agent starts. These are general cluster compatibility fixes.
Network access remains disabled, matching the task's default.

All models use BF16, native 131072 context, high reasoning, temperature 1,
seed 123, output cap 16384, stock 30-message and 210-second tool limits,
a common 7200-second safety timeout, and no cumulative token budget. Dataset:
`princeton-nlp/SWE-bench_Verified`, test split, revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Each model receives all 500 tasks,
shuffled with the same seed, divided into four 125-task shards (two concurrent).
Each shard requests one GPU, 16 CPUs, 256G RAM and 24 hours.

The three-task gold/negative smoke gate validates the stock scoring path on
Django14155, SymPy13878 and Matplotlib20859. It is not an all-task environment
validity claim. Known residual offline-network/test-environment failures are
retained and disclosed; absolute scores are not compared with the model card.

```bash
sbatch src/experiments/training_against_role_probes/swe_standard/controls.sbatch
sbatch --array=0-3%2 --dependency=afterok:GATE_JOB \
  src/experiments/training_against_role_probes/swe_standard/collect.sbatch REPORT_ROOT
sbatch --dependency=afterok:COLLECTION_JOB \
  src/experiments/training_against_role_probes/swe_standard/assemble.sbatch REPORT_ROOT COLLECTION_JOB
sbatch --dependency=afterok:ASSEMBLY_JOB \
  src/experiments/training_against_role_probes/swe_standard/score.sbatch REPORT_ROOT
.tools/bin/uv run --frozen --no-sync python \
  src/experiments/training_against_role_probes/swe_standard/report.py --require-tasks
```

Do not repeat these submissions when the manifest already contains active jobs.
Model roots, export revisions and all dependency jobs are recorded in the
pipeline and per-shard manifests. Original checkpoints and exports are reused.
Large attempts: `MODEL_ROOT/runs/swe_inspect_standard/COLLECTION/shardN/swe_bench/`.
Assembled histories: `MODEL_ROOT/data/swe_inspect_standard/evaluation/`.
Small task results: `MODEL_ROOT/post_training/swe_inspect_standard_task_results.json`.
Whole-history metrics: `MODEL_ROOT/predictions/swe_inspect_standard/summary.json`.
Per-token outputs remain on cluster storage; summaries and report tables are in Git.

Assembly requires all 500 unique attempts, matching settings, and valid exports.
The report checks task counts, dataset identity, input hashes and probe coverage.
No failed model attempt is regenerated. Whole-history role scoring uses the existing
model-specific layer-12 probes; histories beyond native context are excluded only
from role metrics. Task results are reported before role scoring finishes.

[Evaluation code](../../../src/experiments/training_against_role_probes/swe_standard/)
