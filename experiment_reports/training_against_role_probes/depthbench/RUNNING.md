# Reproduction

Source: [TIER / DepthBench](https://github.com/anaykulkarni/TIER), commit
`7e309ca7b651a02b5ee23d2b208ff360b1ff0d9a`. Dataset/split hashes and exact training
IDs are in [data_manifest.json](data_manifest.json). Upstream code/data and
rollouts are retained under ignored `data/` and `runs/` on cluster storage.

```bash
git clone https://github.com/anaykulkarni/TIER.git experiment_reports/training_against_role_probes/depthbench/data/upstream
git -C experiment_reports/training_against_role_probes/depthbench/data/upstream checkout 7e309ca7b651a02b5ee23d2b208ff360b1ff0d9a
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/prepare_corpora.py
.tools/bin/uv sync --locked --project src/experiments/training_against_role_probes/collection_runtime
.tools/bin/uv pip install --python src/experiments/training_against_role_probes/collection_runtime/.venv/bin/python \
  --no-deps -e . -e experiment_reports/training_against_role_probes/depthbench/data/upstream
sbatch --chdir=/data/cluster/users/kortukov/role_probe_training \
  src/experiments/training_against_role_probes/collect_depthbench.sbatch
```

Collection job **1501281**, one B200, sixteen CPUs, 128 GB host RAM.
Logs: `../logs/depthbench-data-1501281.out`; source outputs:
`runs/1501281/{smoke,collection}/calls_0/` through `calls_6/`.
The seven-task smoke requires explicit reasoning before production collection.

The vLLM server uses the existing isolated `server_current` lock (vLLM0.32),
original MXFP4 GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`,
32768 maximum context, Triton attention/MoE, 16 concurrent sequences.
Sampling: temperature1, medium reasoning, 8192 new tokens per assistant call,
24 assistant-turn limit. Seed = 9100000 + task_id*10000 + attempt*100 + turn.
20 attempts/task. All raw generation responses, errors and limits are retained.

The adapter follows the upstream chain-enabled training partition: the first50
zero-call tasks plus all call-bearing tasks whose tools are in the upstream
training function set. It yields 944 tasks. The upstream test loader subsequently
renames GetLocation to GetCurrentLocation; we use no test tasks. Tools are the
relevant set plus3 seeded distractors from the training tool pool; this adapts the
upstream default distractor count to interactive native function calling. A fresh
copy of local tool state is used per rollout. Reference calls/answers are not
sent to the model. We do not claim to reproduce TIER's original reward protocol.

After collection, explicitly verify all task/attempt identities and combine:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble_depthbench.py \
 --collection experiment_reports/training_against_role_probes/depthbench/runs/1501281/collection \
 --error-audit experiment_reports/training_against_role_probes/depthbench/collection_error_audit.json
sbatch --chdir=/data/cluster/users/kortukov/role_probe_training \
 --job-name=depthbench-probes --time=02:00:00 \
 src/experiments/training_against_role_probes/evaluate_corpus.sbatch \
 --input experiment_reports/training_against_role_probes/depthbench/data/train_1_2_3/trajectories.jsonl \
 --output experiment_reports/training_against_role_probes/depthbench/data/baseline
```

The expected combined size is 12,980. Groups0/4/5/6 stay separate and do not enter
training. Failed task-solving attempts remain included; model API failures must
be audited before assembly. No task-success metric is inferred from probe scores.
Baseline evaluation uses the same full-context BF16 model and five probes as
[ASTRA](../astra_sft_1k/RUNNING.md). No model training is started before reporting
the baseline and receiving [implementation readiness](../AGENT_COMMUNICATION.md).

Completed baseline job **1503678** scored all12,980 verified training histories using
`evaluate_corpus.sbatch`; output: `data/baseline/`. The original collection
remains intact, including its error field. The explicitly audited failure is
`depthbench/364/attempt17`; [collection_error_audit.json](collection_error_audit.json)
records exact record/error hashes. Assembly permits only these reviewed failures,
requires exact coverage, and rejects new or modified failures. See
[collection_manifest.json](collection_manifest.json) for all group/output hashes.

The malformed generated tool name contains literal Harmony control tokens.
Retrospective replay preserves it; labels still come exclusively from structured
message/channel metadata, and header/control tokens are excluded from scoring.
No final response is fabricated for the aborted history.

Generate the compact metrics and appendix:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/summarize_corpus.py \
 --baseline experiment_reports/training_against_role_probes/depthbench/data/baseline \
 --report experiment_reports/training_against_role_probes/depthbench \
 --expected-trajectories 12980
```

[baseline_submission.json](baseline_submission.json) records source/probe hashes;
[baseline_manifest.json](baseline_manifest.json) records model/replay settings.
All histories use full context; no truncation, segmentation or downsampling.

## Model training

Job **1503688** was submitted after the complete baseline was reported and the
implementation agent explicitly released the pipeline. Configuration:
[depthbench_config.json](../../../src/experiments/training_against_role_probes/depthbench_config.json).
Provenance: [training_submission.json](training_submission.json). Two B200 GPUs,
sixteen CPUs, 128 GB host RAM, 24-hour limit.

One epoch uses all12,980 histories:812updates, effective batch16 except the final
partial batch, initial microbatch2 per GPU, learning rate5e-5, KL coefficient16.
After each model update, the probe is refitted for2,000steps on the original
C4/Dolma passages, never DepthBench. The common800 KL anchors and eight-task
AppWorld dev corpus are reused; this is cross-domain dev evaluation, not held-out
DepthBench validation. Groups0/4/5/6 do not enter model-update training.

```bash
sbatch --chdir=/data/cluster/users/kortukov/role_probe_training \
 --job-name=depthbench-role-training --time=24:00:00 \
 src/experiments/training_against_role_probes/train.sbatch \
 src/experiments/training_against_role_probes/depthbench_config.json
```

Logs: `../logs/depthbench-role-training-1503688.out`. Checkpoints and rank logs:
`runs/training/epoch1/`.
[W&B run](https://forge.coreweave.com/wandb/kortukov/training_against_role_probes/runs/depthbench-l12-epoch1).

[training_manifest.json](training_manifest.json) captures actual input hashes,
training source hashes, GPU type and runtime version from the running job.

## Fresh-rollout evaluation

The final adapter is evaluated on AppWorld `test_normal` (168 tasks), τ² retail
(114 tasks) and SWE-bench Verified (500 tasks). Jobs and dependencies are in
[post_training/jobs.json](post_training/jobs.json). The root for these commands is
`experiment_reports/training_against_role_probes/depthbench`. Generation and scoring are separate jobs; all failures remain in the cohort.

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
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/validate.py --native-swe
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/compare.py --native-swe
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/refresh_base_reports.py --native-swe
```

`post_training/comparison.json` in the parent experiment consolidates all four
models. Base AppWorld/retail histories and task scores are reused from
`experiment_reports/agentic/`. SWE uses the completed matched native-context
cohorts for all four models. Original base probes and independently fitted
standard-data probes for the trained models are reused. See the
[shared SWE reproduction guide](../RUNNING.md#corrected-swe-evaluation).

No probe evaluation chunks or truncates text. Histories above the native 131,072-token
limit are explicitly unscorable, while task metrics still include them. Current
retail NPZs and `summary.json` live in `predictions/post_training_whole/tau2_retail/`;
AppWorld lives in `predictions/post_training/appworld_test/`; current SWE outputs
live in `predictions/swe_native/`. Pass `--output`
explicitly for retail. Partial earlier scoring outputs are not report inputs.
Recorded per-call contexts remain in the source `generations.jsonl` files; whole
retrospective replay includes past reasoning that the native generation formatter
can omit. The retail-length audit is reproduced by `post_training/audit_retail.py`
in the same experiment-code directory.

The GPU-only extraction jobs use unique node-local `/tmp` scratch directories.
Node `vca-gpu-0701-05` works for this scoring but still fails benchmark-container
startup; exclude it from new container-based generation. Node `vca-gpu-0701-03`
still fails CUDA initialization and remains excluded throughout.
