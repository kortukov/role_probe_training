# Reproducing the AppWorld KL-data comparison

## Training and data

Both variants start from base GPT-OSS-20B, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, dequantized to BF16. They use the
original 2,700 AppWorld train trajectories, one epoch (169 updates), layer-12
post-attention-normalized pre-MLP activations, LoRA rank 64/alpha 128, effective
batch 16, learning rate 5e-5, and KL coefficient 16. Training uses two B200s.
Standard probe refits retain all 250 C4/Dolma passages, the original passage
split, all content tokens and 2,000 SGD steps. No evaluation histories enter KL.

- **Fixed 800:** seed 42 selects 800 base-model training rollouts without outcome
  filtering. Each update independently samples 16 anchors from that fixed pool.
- **Same batch:** KL uses exactly the AppWorld trajectories in the current
  role-loss batch: 16 histories, or 12 in the final update. Startup verifies
  identical trajectory IDs and token IDs between the role and KL inputs.

Both score KL(base || adapted) at unshifted model-authored content positions:
reasoning, answers and tool-call arguments. Tool responses, system/user messages
and synthetic schemas provide context. Complete histories are retained; the
original Dolci KL pool's 2,048-token cap is removed. Thus data identity and retained
context change together. These are fixed base-model histories, not fresh rollouts
from the evolving model. Replay matches the role-training representation, not
necessarily the exact serving prompt bytes.

Selection, input hashes and lengths are recorded in the
[fixed-800 manifest](agentic800/anchors_manifest.json) and
[same-batch manifest](same_batch/anchors_manifest.json). Canonical role-training
input: `experiment_reports/training_against_role_probes/data/train2700/trajectories.jsonl`.
Full-history anchor files are `kl_iteration/VARIANT/data/anchors.jsonl` relative
to the parent experiment. Maximum lengths are 33,649 and 35,505 tokens respectively.

[Training configs and entry points](../../../src/experiments/training_against_role_probes/kl_iteration/).
The locked runtime is shared with the [original experiment](../RUNNING.md).
Prepare fresh output folders using:

```bash
.tools/bin/uv sync --locked --project src/experiments/training_against_role_probes/runtime
sbatch src/experiments/training_against_role_probes/kl_iteration/prepare.sbatch agentic800
sbatch src/experiments/training_against_role_probes/kl_iteration/prepare.sbatch same_batch
```

Run `train.sbatch` with `kl_iteration/agentic800_config.json`, or
`kl_iteration/train_same_batch.sbatch` with `kl_iteration/same_batch_config.json`.
The latter invokes an isolated trainer whose objective change is selecting the
current role batch for KL. Both actual runs first completed two-update pilots,
then resumed `checkpoints/step00002` to finish 169 updates total. Final checkpoints
are `VARIANT/runs/training/epoch1/checkpoints/step00169`. Per-run `manifest.json`
records code/input hashes, model revision, configuration and hardware. Training
jobs 1504830 and 1504852 completed successfully in approximately five hours each.

## Evaluation and canonical sources

Use the existing model export and independent probe-fit scripts under
`src/experiments/training_against_role_probes/post_training/`. Fresh probes are
fitted from scratch with the original 250 passages, original split, all content
tokens, C=0.005 and 5,000 maximum iterations. The refitted training-loop probe is
not used for benchmark reporting. Fresh-probe jobs 1504835/1504854 completed.

AppWorld normal test uses 168 tasks, retail 114, and SWE Verified 500. AppWorld
and retail retain the original evaluation settings. SWE uses the corrected
131,072-token serving context, high reasoning, temperature 1, 16,384 output tokens
per call, 200 messages, ten million cumulative tokens, 3,600 seconds per task and
four concurrent tasks. Baseline and original-model SWE were regenerated using
these settings; their AppWorld/retail cohorts are reused. See the parent report
for precision and role-coverage limitations.

Canonical collection jobs by shard:

| Variant | AppWorld shards 0 / 1 | Retail shard 2 | SWE shards 0 / 1 / 2 / 3 |
|---|---|---|---|
| Fixed 800 | 1505894 / 1505884 | 1504836 | 1505886 / 1505895 / 1505886 / 1505886 |
| Same batch | 1504855 / 1505885 | 1504855 | 1505887 / 1504856 / 1504856 / 1504856 |

Source artifacts live under `VARIANT/runs/post_training/JOB/shardN/BENCHMARK/`
and `VARIANT/runs/swe_native/JOB/shardN/swe_bench/`. Per-variant
`post_training/shard_sources.json` and `post_training/swe_native_shard_sources.json`
identify replacements explicitly. Replacements cover infrastructure failures
before any model generation, never failed task outcomes. Original failed artifacts
are retained locally for audit. Container account lookup failed on node 0702-02;
shared compile-cache collisions and transient imports also caused startup failures.
Exclude 0701-03, 0701-05 and 0702-02 from generation. Replacement servers use
job-local caches; bounded startup retries do not retry benchmark tasks.

## Scoring, validation and publication

Canonical inputs are `VARIANT/data/post_training/BENCHMARK/trajectories.jsonl`;
SWE points to `VARIANT/data/swe_native/evaluation/`. Predictions live under
`VARIANT/predictions/post_training/BENCHMARK/`. Every scored history is replayed
whole. Histories above 131,072 tokens are unscorable for role metrics and remain
in task denominators. No evaluation history is truncated or chunked.

Retail and AppWorld can assemble and score independently of SWE. Assembly accepts
an existing cohort only if it matches byte-for-byte. The finalizer checks input,
model and probe hashes, cohort sizes, metrics and coverage. The independent
`kl_iteration/validate.py` also checks every saved prediction against per-history
confusion counts and exact scored/excluded coverage, plus sampled probabilities.
Baseline and original-model cohorts passed their whole-history validator, including
all corrected SWE attempts.

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/finalize.py --root experiment_reports/training_against_role_probes/kl_iteration/agentic800
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/finalize.py --root experiment_reports/training_against_role_probes/kl_iteration/same_batch
```

Final report jobs are 1504840/1504859; publication job 1504908 waits for both to
succeed, revalidates, and pushes report Markdown and aggregate numeric metrics to
GitHub `kortukov/role_probe_training`, branch `main`, as approved on 2026-10-25.
It excludes weights, raw trajectories and task text. Large artifacts remain under
`/data/cluster/users/kortukov/role_probe_training/experiment_reports/training_against_role_probes/kl_iteration/`.
The published per-variant `publication_metrics.json` contains aggregate task/role
metrics and coverage; detailed confusion matrices are in each `APPENDIX.md`.
