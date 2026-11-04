# Reproduction

The authoritative [configuration](../../../src/experiments/training_against_role_probes/epoch_refresh/off_policy.json)
sets ten epochs, one rollout per training task, batch size 16, learning rate
5e-5, KL coefficient 16, seed 123 and 2,000 probe-refit steps per update.
Training generation allows 8,192 tokens per response, 32,768 context tokens and
50 agent steps, with collection concurrency 8. Standard-probe training retains
250 passages, all content tokens and the original passage split.

[Trainer](../../../src/role_probe_training/epoch_training/train.py),
[matched task scheduler](../../../src/role_probe_training/epoch_training/data.py),
[submission](../../../src/experiments/training_against_role_probes/epoch_refresh/submit.py),
[evaluation job](../../../src/experiments/training_against_role_probes/epoch_refresh/evaluate.sbatch),
[report aggregation](../../../src/experiments/training_against_role_probes/epoch_refresh/report.py).

```bash
.tools/bin/uv sync --locked
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/epoch_refresh/submit.py
# Review the dry-run job plan, then submit from a frozen source snapshot:
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/epoch_refresh/submit.py --submit
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/epoch_refresh/report.py --root experiment_reports/training_against_role_probes/off_policy_epochs
```

Each run uses one B200, eight CPU cores and 256 GiB RAM, with one resumable
48-hour training job per epoch. Every update saves a recovery checkpoint;
epoch checkpoints are retained under `runs/training/epochs/epochNNN`.
Each epoch evaluation runs independently after its training job, retaining
saved retail trajectories and avoiding regeneration when rescoring probes.
Epoch 0 uses the same export, probe fitting and retail protocol as later epochs.

Retail uses temperature 0, seed 123, 16 concurrent tasks, 16,384 output tokens
per response, a 500,000 token task budget and 1,800 seconds per task. Generation
context is 32,768; role replay uses the model's native 131,072-token limit and
explicitly excludes longer histories. Customer simulator: `openai/azure/gpt-4.1`.
Required Azure credentials are inherited without recording their values.
W&B training logs default to offline and are retained with the run.

Large rollouts, exported BF16 models, epoch pools and checkpoints live on cluster
storage below this experiment folder and are ignored by Git. Their manifests,
source/input hashes and retrieval paths are recorded in `jobs.json` and each
run's `manifest.json`. Small validated checkpoint results, fresh probe weights
and metrics remain in Git. Source snapshot and repository/submodule revisions
are recorded at submission and training startup. Do not alter the frozen source
snapshot or the environments while its jobs are active.

Submission status and job dependencies are recorded in [jobs.json](jobs.json). Results appear as checkpoint evaluations complete. Each successful evaluation triggers a CPU publication job that pushes only this experiment’s small validated artifacts to GitHub.
