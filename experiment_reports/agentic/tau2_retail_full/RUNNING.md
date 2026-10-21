# Reproduction

Generation runs and immutable inputs are recorded in [metrics/provenance.json](metrics/provenance.json). Their raw `.eval` logs and `generations.jsonl` retain individual model calls. The consolidated chat data is `data/trajectories.jsonl`; its hash is in [metrics/summary.json](metrics/summary.json). Data is kept on cluster storage and excluded from Git.

Combine completed disjoint shards:

```bash
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/agentic_eval/finalize.py --benchmark tau2_retail --run experiment_reports/agentic/tau2_retail_full/runs/1459495 --selection-manifest experiment_reports/agentic/tau2_retail_full/runs/1459495/manifest.json --output experiment_reports/agentic/tau2_retail_full
```

## Probe predictions and standalone viewer

See [probe reproduction](../probe_transfer/RUNNING.md) and [layer-comparison reproduction](../probe_layer_comparison/RUNNING.md). The shared `agentic_probes/report.py` script accepts repeated `--dataset NAME=PREDICTION_DIRECTORY`, optional benchmark summary paths, and layer choices. It embeds all data in a standalone `viewer.html`, which stays local and can be downloaded and opened without a server.
