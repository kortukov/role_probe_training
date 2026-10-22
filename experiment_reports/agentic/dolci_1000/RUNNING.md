# Reproduction

See `src/experiments/agentic_eval/` for benchmark/download scripts and separate locked environments; `src/experiments/agentic_probes/` for inference and reports.

The shared data format is JSONL with `messages`, tool metadata, IDs, scores and source provenance. Raw outputs live under this report tree in ignored `runs/`, `data/`, or `predictions/` directories.

## Download and sample

The source revision, parquet SHA256 values and exact selected indices are recorded in [manifest.json](manifest.json). Download all six pinned parquet shards, then select uniform rows:

```bash
hf download allenai/Dolci-Instruct-SFT-Tool-Use --repo-type dataset --revision dc042846f0f2de0f15eedae3d6ced04223ed47eb --include 'data/*.parquet' --local-dir .cache/dolci-tool-use
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/agentic_eval/dolci.py --seed 123 --samples 1000
```

The selected JSONL is `data/trajectories.jsonl`. Full five-layer predictions are at `../probe_transfer/predictions/dolci/shard-{0,1,2,3}/`; manifests record the jobs and probe hashes. Four B200 shards completed in 3–6 minutes each.

## Standalone viewer

Run `uv run --frozen --no-sync python src/experiments/agentic_probes/report.py --dataset NAME=PREDICTION_DIRECTORY --layers 12 --output REPORT_DIRECTORY`. Repeat `--dataset` for more inputs; use `--layers 4 8 12 16 20` for the separate layer comparison. The generated `viewer.html` embeds all data, needs no server/network, and is excluded from Git.

## Recorded execution settings

Uniform seeded sample of 1,000 from 227,579 rows, seed 123. Pinned source revision dc042846f0f2de0f15eedae3d6ced04223ed47eb. All five source mixtures are represented. Original function metadata preserved; environment maps to tool. No analysis channel is present. This off-policy static corpus has no GPT-OSS task-success metric.
