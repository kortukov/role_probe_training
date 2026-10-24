# Reproduction

## Code and environment

Reusable implementation: `src/role_probe_training/training/`. Entry points,
configurations and Slurm scripts: `src/experiments/training_against_role_probes/`.
Training uses the isolated, locked `runtime/` environment (PyTorch 2.9.1+cu128,
Transformers 4.57.6, PEFT 0.20). Prepare it with:

```bash
.tools/bin/uv sync --locked --project src/experiments/training_against_role_probes/runtime
```

Base model: `.cache/models/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`. Training dequantizes to BF16, uses
FlexAttention with attention sinks, and enables non-reentrant gradient
checkpointing. Compile-cache overflow fails explicitly rather than selecting
quadratic eager attention. Two B200 GPUs hold one model replica each; model
adapter gradients are summed after globally token-normalized accumulation.

Upstream reference: `submodules/training_against_probes`, revision
`9c3f80cc6d6d6fa2a434ca49ffdd150c4223c427`. Relevant files are
`configs_runs/honesty_example.yaml`, `src/finetune.py`,
`src/utils/finetune_utils.py` and `src/probes/build_kl_rollouts.py`.
Each run's `manifest.json` records source hashes, Git revision and input hashes.

## Trajectories and baseline probes

Collection array **1501202** runs seeds 8301–8330 on all 90 AppWorld train tasks,
with up to eight concurrent one-GPU shards. Each shard runs eight worlds in
parallel, temperature 1, high reasoning, 8,192 output tokens per call and a
32,768-token server context. Complete environment and export settings are in
[AppWorld reproduction](../agentic/appworld/RUNNING.md) and the per-shard manifests.
Assembly requires 2,700 unique attempts, exactly 30 per task, and no infrastructure
or export errors. Unsuccessful and execution-limited attempts remain included.

```bash
sbatch --exclude=vca-gpu-0701-03 src/experiments/training_against_role_probes/collect_train.sbatch
sbatch --dependency=afterok:COLLECTION_JOB src/experiments/training_against_role_probes/assemble_train.sbatch COLLECTION_JOB
sbatch --dependency=afterok:ASSEMBLY_JOB src/experiments/training_against_role_probes/score_train.sbatch
sbatch --dependency=afterok:SCORING_JOB src/experiments/training_against_role_probes/report_train.sbatch
```

Current jobs: assembly **1501282**, baseline scoring **1501329**, report **1501382**.
Canonical data: `data/train2700/trajectories.jsonl`; provenance and generation
success rates: `train_corpus.json`. Records preserve roles, reasoning channels,
tool metadata and task/seed IDs. Generation is seeded but concurrent inference
is not guaranteed bitwise reproducible. Collection timing is retained per shard.
The separate 300-trajectory debug corpus is under `data/debug/`.

Baseline scoring uses the original probes at layers 4, 8, 12, 16 and 20, with
full history and no independent windows. Per-token outputs are under
`predictions/train2700/shard{0,1}/`. Each compressed NPZ contains `input_ids[N]`,
`labels[N]`, `layers[K]`, `predictions[K,N]` and `probabilities[K,N,5]`; the adjacent
index maps files to trajectory IDs and records confusion counts. Labels -1
exclude formatting and synthetic tool-schema tokens. Role order is recorded in
`manifest.json`. Small aggregates are in `baseline_probe_results.json`.

## KL and probe-refit data

The fixed KL pool uses the first 5,000 valid first-user prompts from ordinary
`allenai/Dolci-Instruct-SFT`, shuffled with seed 42, selecting 800. Source revision:
`bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`. The selection follows the upstream
loader recipe; identical prompt IDs to the original authors' private local pool
are not established. GPT-OSS generates greedy completions of up to 1,536 tokens.
Exact Harmony prompt/completion token IDs are stored in `data/kl/anchors.jsonl`;
source and generation manifests are adjacent. Generation job **1501243** completed.

KL(base || adapted) scores unshifted completion positions under the upstream
2,048-token combined cap, with coefficient 16 and 16 sampled anchors per update.
Three anchors have no completion positions within this cap and contribute no KL
tokens. The reference pass disables LoRA on the same model; there is no second
teacher model. Vocabulary projections are checkpointed in 64-token chunks.
Reproduction entry points are `prepare_kl.py` and `generate_kl.sbatch`.

Probe refitting uses only the original 250 C4/Dolma passages from
`experiment_reports/role_probes/gptoss20b_baseline/artifacts/inputs/documents.jsonl`,
all five role renderings, all content tokens and the original passage split.
Each refit warm-starts the probe and runs 2,000 SGD steps: learning rate 1e-3,
weight decay 1e-4, 50-step warmup followed by cosine decay, clipping 1. A minibatch
contains 32 passage-role examples with all their content tokens. This adapts the
upstream sequence-level binary probe to token-level multiclass labels.

## Model training and restart

Configuration: `train_config.json`; production job **1501541** runs after the
completed baseline report, with a 12-hour allocation. One epoch gives 169 updates at effective batch 16;
the final batch contains 12 trajectories. Full trajectories are right-padded,
with microbatch 2 per GPU and global content-token normalization. LoRA rank 64,
alpha 128 and dropout 0 target q/k/v/o and fused expert gate/up/down parameters.
AdamW uses learning rate 5e-5, weight decay 1e-4 and gradient clipping 1.

Layer 12 means zero-based post-attention-layernorm output / pre-MLP input.
Eight separate AppWorld dev trajectories from job **1501280** are evaluated
before training, every five updates and at completion. Their path is pinned in
the config; task IDs must be disjoint from training. These are fixed base-model
trajectories, not fresh rollouts of the adapted model.

```bash
sbatch src/experiments/training_against_role_probes/train.sbatch src/experiments/training_against_role_probes/train_config.json
sbatch src/experiments/training_against_role_probes/train.sbatch src/experiments/training_against_role_probes/train_config.json --resume CHECKPOINT_DIRECTORY
```

Outputs: `runs/training/appworld2700/`. Checkpoints contain the adapter, probe,
optimizer and configuration; the latest two complete states are retained.
Checkpoints are written initially, every ten updates and at completion. If a
role/KL microbatch runs out of GPU memory, both ranks discard the partial update
and retry the same examples with microbatch 1. An OOM at microbatch 1 stops
explicitly; it never silently skips or truncates a trajectory.

W&B: [AppWorld training run](https://forge.coreweave.com/wandb/kortukov/training_against_role_probes/runs/appworld2700-l12-epoch1).
Authentication uses the user's netrc with `WANDB_BASE_URL=https://api.forge.coreweave.com`
and entity `kortukov`. Numeric logs include losses, KL, accuracy, gradient norms,
phase timing, memory and dev metrics. Credentials and task text are not logged.

## Creating the report

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/report_training.py \
  --run experiment_reports/training_against_role_probes/runs/training/appworld2700
```

This updates `training_results.json`, the main report and appendix confusion
matrices. Job **1503673**, dependent on training exit, runs `report_training.sbatch`
and pushes only these small report files. Data, predictions, logs, W&B local
files and checkpoints are ignored by Git. A standalone trained-model viewer is
not generated by this training pipeline.

## Fresh-rollout evaluation

The final adapter is evaluated on AppWorld `test_normal` (168 tasks), τ² retail
(114 tasks) and SWE-bench Verified (500 tasks). Jobs and dependencies are in
[post_training/jobs.json](post_training/jobs.json). The root for these commands is
`experiment_reports/training_against_role_probes`. Generation and scoring are separate jobs; all failures remain in the cohort.

The exported BF16 model folds in expert parameter deltas and retains the original
FP32 attention LoRA separately. `export_model.py` checks layer-12 activations
against the original adapter model before saving. The custom vLLM worker applies
the FP32 attention adapters at inference; probe scoring loads the same adapters
through PEFT. Use one GPU per server (no tensor/pipeline parallelism).

Environments are the existing `agentic_eval/server_current` (vLLM),
`agentic_eval/swe` (Inspect), and `training_against_role_probes/runtime` (activation extraction/scoring), and the
root `.venv` (cuML fitting in a separate CUDA-13 process). Standard-data activations
are retained under `data/post_training/probe_features/`.
Install this package in the server environment without changing its dependencies:

```bash
.tools/bin/uv pip install --python src/experiments/agentic_eval/server_current/.venv/bin/python --no-deps -e .
```

Entry points are in `src/experiments/training_against_role_probes/post_training/`:
`export.sbatch CHECKPOINT OUTPUT`, `probes.sbatch fit --root ROOT`,
`collect.sbatch ROOT MODEL_NAME`, `assemble.sbatch ROOT COLLECTION_JOB_ID`,
`whole_score.sbatch ROOT --overlength exclude`, then `report.sbatch ROOT`. Export must finish before fitting
and collecting; scoring requires both fitting and assembly. Collection array
indices 0–1 are AppWorld, 2 is τ² retail, and 3–6 are SWE-bench. Each server uses
one B200, 32,768-token context and at most 16 active sequences. Inspect runs eight
AppWorld/SWE tasks or 16 retail tasks concurrently per server. Seed is 123 and
reasoning effort is high. AppWorld/SWE use temperature 1 and 8,192 output tokens
per call; retail uses temperature 0 and 16,384 tokens. Task time limit is 1,800s.
AppWorld allows 50 steps, SWE 200 messages, and retail its native 100-turn cap.
Retail uses the authorized Azure GPT-4.1 customer simulator inherited from the
submission environment. Exact settings and source hashes are in each run manifest. Exclude
`vca-gpu-0701-03,vca-gpu-0701-05` when submitting these jobs (see the cluster guide).
If infrastructure requires whole-shard replacement, `post_training/shard_sources.json`
records the source job for each replaced shard; assembly validates full task coverage.

Fresh probes use all content tokens from the original 250-passage source at
`experiment_reports/role_probes/gptoss20b_baseline/artifacts/inputs/documents.jsonl`,
up to 1,024 tokens per passage rendering, with the original passage split.
The standard cuML multinomial logistic fit uses C=0.005, 5,000 maximum iterations
and no feature scaling. No training-loop probe weights initialize this fit.
Probe weights, metadata and held-out passage metrics are under `post_training/`.

Large local artifacts: `data/post_training/model/` contains the exported model;
`runs/post_training/COLLECTION_JOB_ID/` contains Inspect logs and shard exports;
`data/post_training/BENCHMARK/trajectories.jsonl` contains complete canonical
cohorts; `predictions/post_training/BENCHMARK/` contains per-token NPZ files,
trajectory index and summary. These paths are excluded from Git. Small task and
probe results are published in `post_training/results.json`; the report script
validates coverage and matching input/model hashes before updating report tables
and percentage-only appendix matrices. The current four-model report is validated and published together; earlier automatic publisher jobs were cancelled.
Raw benchmark artifacts use the same
message schema as the existing agentic evaluations.

## Current whole-history comparison

The main report is regenerated only after all model/benchmark cohorts are ready:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/validate.py
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/compare.py
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/refresh_base_reports.py
```

`post_training/comparison.json` in the parent experiment consolidates all four
models. Baseline task results and histories are reused from `experiment_reports/agentic/`;
its original five probes are reapplied with whole contexts, without new generation.
The three trained models use their separately generated cohorts and fresh standard-data
layer-12 probes. The original base SWE run has 226 API-error attempts and different
settings; its task score is retained with this limitation, not treated as a controlled
training comparison.

No probe evaluation chunks or truncates text. Histories above the native 131,072-token
limit are explicitly unscorable, while task metrics still include them. Current
retail NPZs and `summary.json` live in `predictions/post_training_whole/tau2_retail/`;
AppWorld and SWE live in `predictions/post_training/BENCHMARK/`. Pass `--output`
explicitly for retail. Partial earlier scoring outputs are not report inputs.
Recorded per-call contexts remain in the source `generations.jsonl` files; whole
retrospective replay includes past reasoning that the native generation formatter
can omit. The retail-length audit is reproduced by `post_training/audit_retail.py`
in the same experiment-code directory.

The GPU-only extraction jobs use unique node-local `/tmp` scratch directories.
Node `vca-gpu-0701-05` works for this scoring but still fails benchmark-container
startup; exclude it from new container-based generation. Node `vca-gpu-0701-03`
still fails CUDA initialization and remains excluded throughout.

Recompute the existing-data SWE failure audit with `uv run python src/experiments/training_against_role_probes/post_training/audit_swe.py`; it reads existing trajectories only and records their hashes in `post_training/swe_failure_audit.json`.

## Corrected SWE evaluation

The native-context rerun is separate from the published constrained cohort until
all models finish and their new trajectories are probed. Its settings and job IDs
are recorded in [post_training/swe_native.json](post_training/swe_native.json).
The base is exported without updates to BF16 by `post_training/export_base.py`,
matching trained expert precision. Existing trained model exports and independently
fitted probes are reused.

`post_training/swe_native.sbatch ROOT pilot` runs six diagnostic tasks;
`sbatch --array=0-3 post_training/swe_native.sbatch ROOT full` covers all 500 tasks
in disjoint shards. Both use 131,072 context tokens, high reasoning, temperature 1,
16,384 output tokens per call, 200 messages, ten million cumulative tokens and
3,600 seconds per task, with four concurrent samples. Increasing the cumulative
budget avoids imposing the previous two-million-token ceiling when longer inputs
are repeatedly sent. The native context still bounds individual calls; exceeding
it is retained as a task outcome and explicitly counted in exported metrics.

Before tasks run, each server must accept a 40K-token request and pass tool-call
checks. `post_training/validate_swe_scorer.py` verifies gold and unchanged-patch
controls on two repositories through the pinned images and upstream grader.
Completed native cohorts are assembled with `post_training/assemble_swe_native.py --root ROOT --collection-id JOB` into `ROOT/data/swe_native/evaluation/`.
Whole-history probe extraction uses that new input and a separate prediction
folder; no old cohort is overwritten or combined with pilot attempts.

After all four native cohorts finish, `validate.py --native-swe` and `compare.py --native-swe` validate and replace the SWE rows and appendix matrices together. The dependent report job updates local reports only; inspect them before committing the new results.

### Session wake-up

`post_training/watch_swe.sh THREAD_UUID` is a one-shot background watcher using
`sacct`, `sleep`, and `codex queue`. It checks every minute and queues a message
back to the specified existing Codex session on pipeline completion, failure, or
a 30-minute progress timeout. The resumed session reviews/pushes ready results,
handles failures, or rearms the watcher. It requires the Codex daemon to remain
available. Local PID/thread/log details are in `logs/swe-watcher.json`; stop an
armed watcher by terminating that PID. Do not start duplicate watchers.
