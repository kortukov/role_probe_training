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

Training has not started. It requires the completed baseline report and the
implementation agent's explicit READY signal in [AGENT_COMMUNICATION.md](../AGENT_COMMUNICATION.md).

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
