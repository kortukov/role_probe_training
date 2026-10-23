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

The production alternating training loop has not been launched: [DESIGN.md](DESIGN.md)
records the concrete adaptations awaiting discussion. W&B project: `training_against_role_probes`; authentication is verified below.
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

## Two-GPU throughput calibration

Job **1493850** completed on two B200 GPUs in 9 minutes 28 seconds. The isolated runtime preserves the environments
used by evaluation jobs. Reproduce with:

```bash
.tools/bin/uv sync --locked --project src/experiments/training_against_role_probes/runtime
sbatch src/experiments/training_against_role_probes/throughput.sbatch
```

Implementation: `src/role_probe_training/training/throughput.py`. Outputs are
`data/throughput/1493850/{smoke,4096,8192,16384,0}/rank{0,1}.jsonl`; zero denotes
full trajectories. Logs are under `logs/role-throughput-1493850.out`.

This is a **role-loss-only calibration**, not end-to-end alternating training.
The fixed layer-12 probe was fitted on standard role-confusion data; no probe is
fitted on AppWorld. KL and standard-data refitting overhead remain to be measured.
The effective batch is 16 trajectories, with microbatch one and token-weighted
accumulation over two replicas. Each condition resets the model/adapters, uses
seed 123 and identical trajectory batches, and runs one warmup plus two measured
updates. A separate longest-trajectory backward tests peak-length feasibility.
The short smoke phase first checks loading, backward and optimizer execution.

Configuration: BF16 dequantized base, FlexAttention with attention sinks, non-reentrant
gradient checkpointing, LoRA rank 64 / alpha 128 / dropout zero on q/k/v/o and
fused expert gate/up and down parameters, AdamW learning rate 5e-5 and weight decay
1e-4, gradient clipping at 1. Loss averages labeled content tokens; formatting and
synthetic tool-schema tokens are excluded. This token weighting is a calibration
choice, not a final decision about role balancing. Probe capture is zero-based
layer 12's post-attention-layernorm output. The backbone runs without vocabulary
logits. No adapter checkpoint is promoted to a trained model.

Reported times must distinguish warmup/compilation, ordinary updates and the
longest-example stress check. Both ranks record peak allocated/reserved memory,
loss, accuracy, gradient norm and elapsed time. W&B receives numeric metrics only;
local JSONL remains the result source if logging is unavailable. These brief loss
traces do not establish convergence or comparative model quality.

A full pass over 27,000 trajectories at effective batch 16 would take 1,688
updates. Context-dependent epoch estimates must use measured timings and add KL,
standard-data activation extraction, probe optimization and validation separately.
No end-to-end throughput estimate is available yet.

### Truncation retains some multi-turn context but changes the data

In debug data, first 4096 tokens retain a median 4 complete tool replies; 96.3% retain
at least two. First 8192tokens retain a median 14 complete tool replies; 99.3% retain
at least two. Only 12% of full trajectories fit 8192 tokens; 71.3% fit 16384. Prefix
truncation therefore drops later interactions and often task completion. These
counts use complete tool-message content spans, not token-budget guesses.

## Full-trajectory microbatch comparison

Job **1500991** completed in 10 minutes 51 seconds, comparing microbatches 1, 2 and 4 per GPU on two B200s in the
short-job `ber-testing` partition. All conditions retain effective batch 16 and
use the same five seeded batches, sorting trajectories by length within each
batch and alternating ranks. One update is warmup; four are measured. Right
padding is masked out of attention and loss. A separate stress test repeats the
longest trajectory across the entire microbatch.

Before timing, compare token-weighted cross-entropy and all adapter gradients
from individually processed versus jointly padded examples, without updating
weights. The four seed-selected prefixes contain 2048/1536/1792/1024 tokens.
Calibration gates are loss relative error <1%, gradient cosine >0.99 and gradient
relative L2 error <10%; these are practical checks, not universal guarantees.
The existing Transformers FlexAttention implementation is unchanged. Hidden
activations are not required to be bitwise identical across batch shapes.

```bash
sbatch src/experiments/training_against_role_probes/microbatch.sbatch
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/summarize_microbatch.py \
  --runs experiment_reports/agentic/training_against_role_probes/data/microbatch/1500991 \
  --output experiment_reports/agentic/training_against_role_probes/microbatch_results.json
```

The paired comparison reruns microbatch 1 with the same ordering and explicit
attention mask as larger batches. Use this paired baseline for speedups, rather
than the earlier context experiment. Local outputs: `data/microbatch/1500991/`;
log: `logs/role-microbatch-1500991.out`. KL and probe refitting remain excluded.
Automatic OOM fallback is intentionally disabled in this fixed-size calibration.
For production, consider conservative padded-token budgets and coordinated
whole-update retries with smaller microbatches; this is not yet implemented.

Small validated aggregates are `microbatch_results.json` and `context_results.json`.
The original context run used source commit `166ff7b`; the current benchmark adds
explicit padding masks and length grouping for the paired microbatch comparison.
All three conditions and both ranks completed; no OOM or retry occurred.
