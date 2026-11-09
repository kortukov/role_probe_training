# Reproduction

Training job: `1766203`. [jobs.json](jobs.json) records the exact submission command, inherited environment, configuration path and source hashes. To reproduce, run that command with the recorded environment from the repository root after preparing the existing uv runtime. Do not resubmit over an active output directory.

Configuration: `src/experiments/training_against_role_probes/on_policy_variants/batch64.json`. Entry point: `role_probe_training.on_policy_variants.train`. This separate variant trainer preserves the sources used by the queued 2,700-episode run; all submitted source hashes were verified unchanged. This code was removed from the tree; it is archived at commit `2e475d0`.

Each job requests two GPUs, 16 CPUs, 256 GB RAM and 12 hours on ber-gpu1. The prepared uv training runtime and separate vLLM runtime match the [2,700-episode run](../appworld2700/RUNNING.md). Learning rate is 5e-5, KL coefficient 16, seed 123 and probe refit 2,000 SGD steps. Validation runs every five updates and at completion. Final task evaluation and probe scoring are now attached; see EVALUATION.md. GitHub publication is separate.

Large artifacts stay under this experiment's `runs/epoch1/`, logs under `logs/`, and W&B files under `wandb/` on cluster storage. The runtime manifest records source/data provenance and checkpoints retain resume state. Both variants select the first three scheduled occurrences per task from the original shuffled schedule, preserving episode seeds without filtering outcomes. Only task identities and scheduling metadata are reused: fresh full episodes are generated for each update.

Lightweight schedule, syntax and source-integrity checks passed before submission. Existing two-GPU colocation checks supply GPU-path validation.

## Current repaired submission

Job `1787803` replaces `1766203`. See [repaired_jobs.json](repaired_jobs.json) for the exact command, environment and hashes. It uses the separate `on_policy_repaired` trainer and configuration, with output in `runs/epoch1_repaired/`. The original artifacts remain intact. Mixed reasoning/text/tool messages replay their retained original Harmony completion; ordinary messages keep the original template behavior. All 2,844 saved prefixes passed the regression check, including the failing mixed message. The run restarts from base with the same schedule and seeds, without filtering failed episodes. This code was removed from the tree; it is archived at commit `2e475d0`.

## Automatic final evaluation

See [EVALUATION.md](EVALUATION.md) and [evaluation_jobs.json](evaluation_jobs.json) for the submitted dependencies, exact commands and script hashes. The final checkpoint is exported once, then fresh probe fitting and benchmark generation proceed in parallel. Existing prepared uv runtimes are reused. Retail inherits Azure credentials via plain `--export=ALL`; no secrets are stored in the manifest.

Large merged models, probe features and assembled trajectories live under this experiment's `data/`; original logs live under `runs/`, per-token scores under `predictions/`. Fresh probe weights and small result JSON/Markdown files live under `post_training/`. These paths are relative to this experiment on cluster storage. SWE uses the existing native-context Inspect generous snapshot protocol, four shards at concurrency four, with 72 hours per shard; AppWorld and retail retain the existing comparison settings. GPU array concurrency is capped at two per collection family.

Each downstream job requires predecessor success. Infrastructure failures block affected results rather than silently dropping tasks. Result writers verify task counts, trajectory hashes and checkpoint identity. No additional training or GPU pilot was launched for this wiring.

## Current Harmony-parser restart (2026-10-31)

[harmony_jobs.json](harmony_jobs.json) supersedes the earlier training submission; [evaluation_jobs.json](evaluation_jobs.json) records the reconnected evaluations. The run resumes from update 0 into `runs/epoch1_harmony/`, preserving model, optimizer and probe state. A checkpoint-hashed migration verifies original sources and the unchanged task schedule. The two 270-episode runs precede the 2,700-episode continuation in user job priority.

The vLLM worker now invokes the installed vLLM 0.32 HarmonyParser with oss-harmony 0.0.11; no heuristic moves tool names from channels into recipients. Invalid/unsupported model output produces no executable tool call, ends the Inspect tool loop, and is scored on the actual AppWorld state. `parser_events.jsonl` and trajectory `invalid_model_outputs` record these events. Exact sampled tokens remain in `requests.jsonl` for reverse KL; retrospective role scoring covers the returned conversation, not discarded malformed segments. Such segments remain auditable in raw-token records. CUDA/transport/provenance failures remain fatal.

For the interrupted batch, `recovery.json` hashes saved completions. They are replayed only at identical prompt IDs, seed and policy step; mismatch aborts instead of drawing replacement samples. All attempts and failed-run artifacts remain retained. New generations begin after the saved prefix.

CPU regression checks covered both malformed headers, ordinary controls, the full-run failing history, real Inspect episode termination, HTTP isolation of failures, raw-token retention, checkpoint state and schedule identity. No additional GPU pilot was submitted. Test entry points are in `src/experiments/training_against_role_probes/on_policy_harmony/`. This code was removed from the tree; it is archived at commit `2e475d0`.

## Current whole-episode recovery (2026-11-01)

[recovery_jobs.json](recovery_jobs.json) supersedes the failed Harmony restarts. The active output is `runs/epoch1_recovery/`; model/optimizer/probe checkpoints still resume at steps 0, 3 and 2 for batch-64, batch-16 and full training respectively. [evaluation_jobs.json](evaluation_jobs.json) points exports and the existing AppWorld/retail/SWE/probe-scoring DAG to the replacement final checkpoints.

Re-executing saved tool calls changed unordered set output and Python object addresses, so exact prompt replay could not reproduce prior contexts. Recovery now stages audited same-policy trajectories and their exact generation tokens, never re-executing their tools. The short batch-64 run recovers 41 episodes and generates the remaining 23 for its first update; the interrupted batch-16 and full-run updates each recover all 16 episodes. `runs/epoch1_recovery/recovery.json` records cache hashes and source hashes.

Three terminal attempts retain original error records and are labelled `recovered_terminal`: two malformed-model-output attempts and one context-limited attempt. Their task scores come from the original Inspect log's AppWorld final-world metadata, using the same success criterion as the scorer. They are not rerolled or removed. Raw completion tokens remain in reverse KL; retrospective role supervision covers the saved returned conversation, with any unreturned malformed segment retained only in raw-token diagnostics.

Future context exhaustion records `episode_stops.jsonl`, ends that episode with zero generated tokens, and lets AppWorld score the current world state. Nothing is truncated and the server remains usable. Infrastructure/model-compute failures remain fatal. Validation evidence is in [recovery_validation.json](recovery_validation.json); executable checks and submission code are under `src/experiments/training_against_role_probes/on_policy_recovery/`.

## Current single-GPU continuation (2026-11-01)

Training job **1816749** replaces 1816549; export
job **1816750** and all downstream jobs are recorded in
[evaluation_jobs.json](evaluation_jobs.json). The replacement uses one GPU,
eight CPU cores, 256 GiB host RAM and a 24-hour limit. It resumes step
0 and finishes at step 5; the effective episode
batch and objective are unchanged. Current final checkpoint: `experiment_reports/training_against_role_probes/on_policy/appworld270_batch64/runs/epoch1_single_gpu/checkpoints/step00005`.

The common trainer is `src/role_probe_training/on_policy_training/train.py`.
`src/experiments/training_against_role_probes/on_policy_execution/train.sbatch`
sets torchrun process count from `SLURM_GPUS_ON_NODE`, supporting `sbatch --gpus=1`
or `--gpus=2`. Historical queued code remains frozen; no separate one-GPU
training-loop implementation is used. Migration authorizes the world-size change,
and audited recovery episodes/requests from both previous ranks are merged
without rerunning tools. Scripts under `on_policy_single_gpu/` prepare and submit
this particular migration, not a separate trainer.

Validation: CPU single-rank collective, gradient-normalization and probe-refit
checks passed; real cached collection recovered all 16 batch-16 episodes without
model generation. Batch-64 cache integrity/coverage passed (41 cached,23 fresh).
GPU execution and single-GPU throughput remain unverified at submission.

## Concurrent-collection comparison (2026-11-03)

Training job **1964052** (commit `043ff56`) repeats this run from base, steps 0 to 5, on one GPU
with 16 CPUs and `collection_concurrency: 8`. Schedule, episode seeds, objective and serving match
the single-GPU run. Output: `runs/epoch1_concurrent_v2/`; W&B ID suffix `-concurrent8-v2`.
Configuration: `src/experiments/training_against_role_probes/on_policy_concurrent/appworld270_batch64.json`,
launched with the shared `on_policy_execution/train.sbatch`. [concurrent_jobs.json](concurrent_jobs.json)
records the job and the failed first attempt.

Previously, the collector ran one Inspect episode at a time against a single-threaded server, so
vLLM, already configured for eight sequences, decoded one request at a time. `on_policy_recovery`
now runs up to `collection_concurrency` episodes (default 1) in fresh worker processes. Each episode
has its own server route, `/episodes/<id>/v1`, so generation indices, request seeds, tool-call IDs
and retained raw Harmony completions stay per episode. Requests are reordered into plan order.
Batched decoding can change floating-point results, so samples are not bitwise identical to
sequential collection; the sampling distribution is unchanged. After a failure, no new episodes
start; running ones finish and are retained.

The first attempt, job 1816756 in `runs/epoch1_concurrent/`, failed request validation at step 1
before any update. Inspect's vLLM provider keeps the first `base_url` per model name for the life of
a process, so reused workers served later episodes under their first episode's route. Each episode
now gets a fresh process, and the server rejects requests whose seed does not match their route.
That attempt's collection still gives a valid speed measurement: 64 episodes and 303k generated
tokens in 593 s (about 511 tokens/s). Fully fresh sequential steps 2–4 needed 2,207–2,462 s
(about 157–163 tokens/s).

CPU checks with real AppWorld worlds, Inspect, tokenizer and Harmony parser, using a scripted
sampler, covered concurrency 4 and concurrency 2 with reused workers. Both confirmed interleaved
requests, exact seeds, indices and tool-call IDs, and request validation. A separate negative check
confirmed the route/seed guard.

Compare once the run finishes:

```bash
.tools/bin/uv run --frozen --no-sync \
  python src/experiments/training_against_role_probes/on_policy_concurrent/compare.py \
  --sequential experiment_reports/training_against_role_probes/on_policy/appworld270_batch64/runs/epoch1_single_gpu \
  --concurrent experiment_reports/training_against_role_probes/on_policy/appworld270_batch64/runs/epoch1_concurrent_v2 \
  --output experiment_reports/training_against_role_probes/on_policy/appworld270_batch64/concurrency_comparison.json
```

The sequential reference's step 1 recovered 41 of 64 episodes from cache, so compare seconds per
fresh episode or tokens per second. No downstream evaluation DAG is attached to this run.
