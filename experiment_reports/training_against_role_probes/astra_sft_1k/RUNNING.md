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
