# Reproduction

Predictions use schema v1, one gzip JSON per trajectory. Each contains original messages, normalized replay messages, token IDs/offsets, labels, score mask, context boundaries and per-layer probabilities and argmax predictions. Input/probe hashes and job metadata are in each prediction directory’s manifest.

Generate the report and standalone viewer from the repository root:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py --dataset 'dolci_tool_use=experiment_reports/agentic/probe_transfer/predictions/dolci' --dataset 'tau2_retail=experiment_reports/agentic/probe_transfer/predictions/retail' --dataset 'swe_bench=experiment_reports/agentic/probe_transfer/predictions/swe_selected' --expected-count 'dolci_tool_use=1000' --expected-count 'tau2_retail=114' --expected-count 'swe_bench=500' --benchmark-summary 'tau2_retail=experiment_reports/agentic/tau2_retail_full/metrics/summary.json' --benchmark-summary 'swe_bench=experiment_reports/agentic/swe_bench_full/metrics/summary.json' --layers 4 8 12 16 20 --output experiment_reports/agentic/probe_layer_comparison
```

Add another `--dataset NAME=PATH` to include a dataset. See `src/experiments/agentic_probes/run.py --help` for inference and `src/experiments/agentic_eval/dolci.py` for seeded sampling. Generated predictions, trajectories and HTML remain on cluster storage; code, manifests and aggregate metrics are versioned.

## Shared inputs and validation

All layers were captured together at `model.model.layers[layer].post_attention_layernorm`; no decoder-block outputs are used. See the [shared prediction manifests](../probe_transfer/metrics/prediction_manifests.json), [SWE prediction selection](../probe_transfer/metrics/swe_prediction_selection.json), and [input/validation instructions](../probe_transfer/RUNNING.md#inputs-and-validation). These are exactly the same 1,614 trajectories as the layer-12 experiment. The report generator verifies all requested layers, coverage counts and saved confusion matrices against token predictions.

Run `bash src/experiments/agentic_probes/final_reports.sh` to rebuild both reports and standalone HTML files. The complete five-layer viewer is approximately 1.73 GB; download that single file and open it locally in a modern browser. It needs no server or adjacent files and decodes one trajectory at a time. Data-filled HTML and per-token predictions remain excluded from Git.
