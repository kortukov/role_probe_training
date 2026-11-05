# Reproduction

The [configuration](../../../src/experiments/training_against_role_probes/same_batch_epochs/config.json)
extends the original same-batch run to five total epochs. It restores
`../kl_iteration/same_batch/runs/training/epoch1/checkpoints/step00169/`,
including AdamW moments and the warm-start probe. Steps 170–845 perform four
additional complete passes. The original checkpoint is never modified.
Learning rate 5e-5, KL coefficient 16, effective batch 16, two GPU ranks,
microbatch 2 per GPU (coordinated OOM fallback to 1), seed 123, 2,000 probe-refit
steps, and the original full histories remain unchanged. Epoch ordering uses
the original seeded shuffle stream, including advancement past epoch 1.

[Continuation trainer](../../../src/role_probe_training/training/train_same_batch_epochs.py),
[training job](../../../src/experiments/training_against_role_probes/same_batch_epochs/train.sbatch),
[submission](../../../src/experiments/training_against_role_probes/same_batch_epochs/submit.py),
[evaluation job](../../../src/experiments/training_against_role_probes/same_batch_epochs/evaluate.sbatch),
[report aggregation](../../../src/experiments/training_against_role_probes/same_batch_epochs/report.py).

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/same_batch_epochs/submit.py
# After reviewing the plan and freezing sources:
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/same_batch_epochs/submit.py --submit
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/same_batch_epochs/report.py --root experiment_reports/training_against_role_probes/same_batch_epochs
```

Use `uv sync --locked` only when preparing a new environment; do not mutate
shared environments used by concurrent jobs. Jobs run from a source snapshot
with `PYTHONPATH` pointing to its copied `src/`, with existing locked uv
runtimes reused. [jobs.json](jobs.json) records the frozen source path, source
hashes, repository/submodule revisions, input hashes, resource requests, and
dependencies. The run manifest additionally records software and hardware.

One two-B200 allocation (eight CPU cores, 256 GiB RAM, 48-hour limit) performs
continuation. Recovery checkpoints rotate separately from retained epoch
checkpoints at `runs/training/epochs/epoch002`, `epoch003`, and `epoch005`.
Each retained checkpoint has a one-B200 evaluation allocation (eight CPU cores,
256 GiB RAM, 60-hour limit) for export, fresh probe fitting, AppWorld/retail
collection, assembly, whole-history scoring, and report update. Failed stages
stop explicitly; inspect saved artifacts before any recovery to avoid duplicate
attempts. All jobs use normal priority and exclude the documented bad nodes.

Generation uses the checked BF16 expert export plus separate FP32 attention
adapters, identical to the original evaluation representation. Serving context
is 32,768 tokens; role replay uses native 131,072 tokens without truncation.
AppWorld uses the shared test_normal runner: temperature 1, high reasoning,
8,192 output tokens per response, 50 steps, concurrency 8. Retail uses
temperature 0, seed 123, concurrency 16, 16,384 output tokens per response,
500,000-token task budget, 1,800 seconds per task, and GPT-4.1 customer simulator.
Azure credentials are inherited and never recorded. W&B defaults to offline.

Epoch-1 results are copied from the original saved task manifests and
whole-history score summaries, with matching input/model/probe hashes checked.
Original paths and hashes are recorded in each small result JSON. No epoch-1
attempt is regenerated. Fresh evaluation probes use C=0.005 and max_iter=5,000
cuML logistic regression on all original standard-passage content tokens;
metadata and extraction/fitting logs retain stopping diagnostics.

Large trajectories, activation caches, exports, checkpoints, and per-token
predictions remain in ignored `runs/`, `data/`, and `predictions/` directories
below this report and its `evaluations/epochNNN/` folders on cluster storage.
Small validated results, probe weights, metrics, and appendix matrices stay
in Git. Re-scoring must reuse saved trajectories. The report updates measured
tables without overwriting its authored interpretation. Results are pending
until these jobs finish. Submitted 2026-11-04: training **2020991**; epoch-2,
epoch-3, and epoch-5 evaluations **2020992**, **2020993**, **2020994**. At launch,
training was pending scheduler priority and evaluations were dependency-blocked.
These are submission-time states; consult Slurm for current status.
