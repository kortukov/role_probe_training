# Reproduction

Source: [Emperorizzis/ASTRA-SFT-1k](https://huggingface.co/datasets/Emperorizzis/ASTRA-SFT-1k),
revision `a154c6f50b685b6248415dfe3cd4c511cf1c4862`. All 1,000 released rows.
See [data_manifest.json](data_manifest.json) for source/output hashes and coverage.
Large data is retained under this folder's ignored `data/` on cluster storage;
no source data or trajectories are committed. Download with:

```bash
hf download Emperorizzis/ASTRA-SFT-1k data.parquet README.md \
  --repo-type dataset --revision a154c6f50b685b6248415dfe3cd4c511cf1c4862 \
  --local-dir experiment_reports/training_against_role_probes/astra_sft_1k/data/source
```

From `/data/cluster/users/kortukov/role_probe_training`, use `.tools/bin/uv`.
The shared preparation entry point also prepares DepthBench and requires that
upstream checkout (see its RUNNING.md):

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/prepare_corpora.py
sbatch --chdir=/data/cluster/users/kortukov/role_probe_training \
  src/experiments/training_against_role_probes/evaluate_corpus.sbatch \
  --input experiment_reports/training_against_role_probes/astra_sft_1k/data/trajectories.jsonl \
  --output experiment_reports/training_against_role_probes/astra_sft_1k/data/baseline
```

Completed baseline job: **1501446**, one B200, eight CPUs, 128 GB host RAM.
Logs: `../logs/corpus-probes-1501446.out`. The root locked uv environment is used.
Model: cached `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, MXFP4 dequantized to BF16.
Full-context flex attention; no windows, truncation, or activation-site changes.
Probes: existing layers 4/8/12/16/20 at post-attention-layernorm output.
Per-trajectory confusion counts and probability sums are retained in
`data/baseline/per_trajectory.jsonl`; run metadata and aggregate metrics are adjacent.

Conversion code: [corpora.py](../../../src/role_probe_training/trajectories/corpora.py).
`reasoning_content` becomes `thinking`; function calls and matched responses
retain names and receive deterministic call IDs. Tool schemas come from
`mcp_info.base_info.tool_list`. They are excluded from metric denominators.
Intermediate assistant speech is commentary; the final assistant response is final.
No inferred reasoning, regenerated answers, or fabricated tool outputs are added.
This retrospective rendering is not a claim of original serving-prompt recovery.

Training job **1503671** was submitted after the complete baseline was reported
and the implementation agent posted EXPLICIT READY in
[AGENT_COMMUNICATION.md](../AGENT_COMMUNICATION.md).

Generate the compact results and appendix (the main report is authored separately):

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/summarize_corpus.py \
  --baseline experiment_reports/training_against_role_probes/astra_sft_1k/data/baseline \
  --report experiment_reports/training_against_role_probes/astra_sft_1k \
  --expected-trajectories 1000
```

[baseline_submission.json](baseline_submission.json) records code/probe hashes;
[baseline_manifest.json](baseline_manifest.json) records replay/model settings.
The complete 1,000-history baseline finished successfully before model training.

## Model training

Configuration: [astra_sft_1k_config.json](../../../src/experiments/training_against_role_probes/astra_sft_1k_config.json).
Submission provenance: [training_submission.json](training_submission.json).
Two B200 GPUs, sixteen CPUs, 128 GB host RAM, four-hour limit.
One epoch uses all 1,000 histories: 63 updates, with an effective batch of16
except the final partial batch. Initial microbatch2 per GPU, learning rate5e-5,
KL coefficient16, 2,000 standard-data probe-refit steps after each model update.
The role probe is refitted on the original C4/Dolma passages, never ASTRA.
The common800 KL anchors and eight-task AppWorld dev corpus are reused; this dev
evaluation is explicitly cross-domain, not held-out ASTRA validation.

```bash
sbatch --chdir=/data/cluster/users/kortukov/role_probe_training \
  --job-name=astra-role-training --time=04:00:00 \
  src/experiments/training_against_role_probes/train.sbatch \
  src/experiments/training_against_role_probes/astra_sft_1k_config.json
```

Logs: `../logs/astra-role-training-1503671.out`. Checkpoints and rank logs:
`runs/training/epoch1/`.
[W&B run](https://forge.coreweave.com/wandb/kortukov/training_against_role_probes/runs/astra_sft_1k-l12-epoch1).

[training_manifest.json](training_manifest.json) captures actual input hashes,
training source hashes, GPU type and runtime version from the running job.

## Fresh-rollout evaluation

The final adapter is evaluated on AppWorld `test_normal` (168 tasks), τ² retail
(114 tasks) and SWE-bench Verified (500 tasks). Jobs and dependencies are in
[post_training/jobs.json](post_training/jobs.json). The root for these commands is
`experiment_reports/training_against_role_probes/astra_sft_1k`. Generation and scoring are separate jobs; all failures remain in the cohort.

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
