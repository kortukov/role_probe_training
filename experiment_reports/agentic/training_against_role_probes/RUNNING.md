# Reproduction

## Debug collection

The collection job reuses the validated AppWorld agent, environment, evaluator,
server and normalized JSONL export. Each shard has its own inference GPU and eight
concurrent worlds. Seeds 7301–7303 collect all 90 training tasks; seed 7304 collects
30 after deterministic shuffling. Sampling temperature is one. Existing 90-task
benchmark trajectories are not silently included in this new corpus.

```bash
sbatch --exclude=vca-gpu-0701-03 src/experiments/training_against_role_probes/collect_debug.sbatch
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble_debug.py \
  --runs experiment_reports/agentic/training_against_role_probes/runs/JOB_ID \
  --output experiment_reports/agentic/training_against_role_probes/data/debug
```

The assembler requires complete error-free exports with 300 distinct attempt IDs
and 90 distinct tasks. Execution-limited and unsuccessful attempts remain included.
An attempt's unique ID includes its original task ID and generation seed. Generation
seeds and task sampling are recorded; concurrent GPU execution is not promised to
be bitwise deterministic. JSONL retains messages, channels, tools and scores.

Large data, logs, checkpoints and W&B local files are ignored. Keep the resulting
small corpus manifest and training metrics in Git. Collection uses the existing
locked `agentic_eval/swe` and `agentic_eval/server_current` environments; see the
[AppWorld reproduction guide](../appworld/RUNNING.md) for exact runtime details.

## Upstream reference and training status

Training source: `submodules/training_against_probes`, revision
`9c3f80cc6d6d6fa2a434ca49ffdd150c4223c427`. Relevant code:
`configs_runs/honesty_example.yaml`, `src/finetune.py` (model/refit loop),
`src/utils/finetune_utils.py` (KL), `src/probes/build_kl_rollouts.py` (anchor).

The role training loop is not implemented or launched yet: [DESIGN.md](DESIGN.md)
records the concrete adaptations awaiting discussion. W&B project intended:
`training_against_role_probes`; verify authenticated account before the first run.
No credentials or task text should be put in tracked config or W&B logs.

## Completed debug collection

Collection array **1492660** and assembly job **1492664** completed successfully.
The debug corpus is `data/debug/trajectories.jsonl`; its checksum and collection
metrics are in `debug_corpus.json`. `debug_submission.json` records the original
collection-code revision and seeds. Original split metadata remains `train` because
these are AppWorld training tasks; the corpus purpose is `debug`.

The planned training corpus is **300 attempts per task × 90 tasks = 27,000**.
It is separate from the debug corpus and has not been launched. The four-shard
script above reproduces debug collection only, not production collection.

Compute debug length/padding statistics without loading model weights:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/batch_stats.py \
  --data experiment_reports/agentic/training_against_role_probes/data/debug/trajectories.jsonl \
  --output experiment_reports/agentic/training_against_role_probes/debug_batch_stats.json
```

## W&B authentication

Authenticated as `kortukov` at `https://api.forge.coreweave.com`; verified via the
server's read-only viewer query. Credentials are in the user's netrc, not the repo.
Set `WANDB_BASE_URL=https://api.forge.coreweave.com`, `WANDB_ENTITY=kortukov` and
`WANDB_PROJECT=training_against_role_probes` in training jobs. No new login is needed.

## Capacity arithmetic (not a training benchmark)

`debug_batch_stats.json` measures actual corpus sizes. `cost_estimate.json` applies
explicit, **unmeasured** throughput assumptions: 2000 role-update tokens/s/GPU and
8000 activation-extraction tokens/s/GPU, ideal data parallel scaling and no padding.
They are sensitivity inputs, not hardware-derived predictions. Halving both rates
doubles the calculated times. Real throughput varies with length, kernels and
adapter targets; benchmark before scheduling production training.

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/estimate_cost.py \
  --stats experiment_reports/agentic/training_against_role_probes/debug_batch_stats.json \
  --output experiment_reports/agentic/training_against_role_probes/cost_estimate.json
```

For one epoch of 27,000 trajectories, effective batch 16 means 1688 model updates.
Each example below retains the beginning of the trajectory up to its cap. Token
counts extrapolate the 300-debug-trajectory distribution by a factor of 90.

| Context | Estimated epoch tokens | Model role updates, 2 GPUs | Model role updates, 4 GPUs |
|---|---:|---:|---:|
| First 4096 tokens | 110.6M | 7.7h | 3.8h |
| First 8192 tokens | 217.8M | 15.1h | 7.6h |
| First 16384 tokens | 336.4M | 23.4h | 11.7h |
| Full trajectories | 368.9M | 25.6h | 12.8h |

These are **role model updates only**. At 16 KL pairs/update capped at 2048 tokens,
KL adds an illustrative 4.8h on 2 GPUs or 2.4h on 4 GPUs, using the same training rate
plus a separate teacher forward at the extraction rate. Shorter KL sequences
reduce this token count. KL does not shrink when AppWorld prefixes are shortened.

Probe refitting needs new activations after every model update. Including initial
fitting, the full 27k corpus would require about 623B token positions of extraction
for one epoch. Under the above rates that alone is 451 days on 2 GPUs or 225 days on4.
This is not a viable default; no change to fitting data has been silently made.

For comparison only, a fixed **hypothetical 300-trajectory training-side fitting
pool**, distinct from the debug corpus, yields the following combined model-role,
KL and extraction times. This pool size is not approved or implemented.

| Context | 2 GPUs | 4 GPUs |
|---|---:|---:|
| First 4096 | 48.5h | 24.3h |
| First 8192 | 90.9h | 45.5h |
| First 16384 | 137.8h | 68.9h |
| Full | 150.6h | 75.3h |

Both tables exclude actual probe optimizer time, validation, checkpoint I/O,
communication inefficiency, loading and trajectory collection. For 1688 refits,
even 10 seconds of probe optimization per phase adds 4.7 hours, and 60 seconds adds 28.1h.
Do not infer end-to-end speed from these tables before measuring those components.

At the debug collection's observed aggregate rollout cost, generating 27k full
trajectories extrapolates to 71.4 GPU-hours: roughly 35.7h with 2 GPUs or 17.9h with 4,
before startup, queue delays and variation in sampled trajectories.

### Memory arithmetic

CPU-only meta-model inspection counts 20,914,757,184 parameters: **38.96GiB** in BF16.
Candidate rank 64 adapters for q/k/v/o plus each expert's fused gate/up and down
projections total 739,639,296 parameters. A conservative 16 bytes/adapter parameter
(weights, gradients, FP32 moments and master copy) is 11.02 GiB. Exact target support
and optimizer layout remain to be validated; attention-only adapters are much
smaller, but would change the requested upstream adaptation.

With microbatch 1 and checkpointing, storing one BF16 hidden input per decoder
layer costs `24 × sequence_length × 2880 × 2` bytes:

| Maximum length | Checkpoint input tensors | One naive BF16 attention-score tensor |
|---|---:|---:|
| 4096 | 0.53GiB | 2GiB |
| 8192 | 1.05GiB | 8GiB |
| 16384 | 2.11GiB | 32GiB |
| 33241 (debug maximum) | 4.28GiB | 131.72GiB |

The attention tensor calculation is `64 × length² × 2` bytes and illustrates
why eager attention is unsuitable; efficient attention must avoid materializing
it. Base + candidate adapters + checkpoint inputs are roughly 50.5–54.3GiB, **not
peak memory**. Expert intermediates, attention backward, logits, temporary buffers
and allocator overhead remain. Role-only passes should avoid full-vocabulary
logits; KL uses its separate short sequences. Whole-corpus layer 12 BF16 activations
would occupy about 1.93 TiB; a 300-trajectory pool about 22 GiB before labels/overhead.

Two or four data-parallel GPUs each hold a model replica: more GPUs increase
throughput but do not divide per-GPU model memory. Full trajectories look feasible
within the cluster's 192GB-class GPUs with efficient kernels and microbatch 1, but
no training backward/peak-memory measurement has verified this yet.

### Truncation retains some multi-turn context but changes the data

In debug data, first 4096 tokens retain a median 4 complete tool replies; 96.3% retain
at least two. First 8192tokens retain a median 14 complete tool replies; 99.3% retain
at least two. Only 12% of full trajectories fit 8192 tokens; 71.3% fit 16384. Prefix
truncation therefore drops later interactions and often task completion. These
counts use complete tool-message content spans, not token-budget guesses.
