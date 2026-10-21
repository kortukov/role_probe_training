# Reproduction

Predictions use schema v1, one gzip JSON per trajectory. Each contains original messages, normalized replay messages, token IDs/offsets, labels, score mask, context boundaries and per-layer probabilities and argmax predictions. Input/probe hashes and job metadata are in each prediction directory’s manifest.

Generate the report and standalone viewer from the repository root:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py --dataset 'dolci_tool_use=experiment_reports/agentic/probe_transfer/predictions/dolci' --layers 12 --output experiment_reports/agentic/probe_transfer
```

Add another `--dataset NAME=PATH` to include a dataset. See `src/experiments/agentic_probes/run.py --help` for inference and `src/experiments/agentic_eval/dolci.py` for seeded sampling. Generated predictions, trajectories and HTML remain on cluster storage; code, manifests and aggregate metrics are versioned.
