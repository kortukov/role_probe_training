#!/bin/bash
# Refresh both metrics/appendices/viewers; preserve authored README and RUNNING files.
set -euo pipefail
args=(
 --dataset tau2_retail=experiment_reports/agentic/probe_transfer/predictions/retail
 --dataset swe_bench=experiment_reports/agentic/probe_transfer/predictions/swe
 --dataset appworld_test_normal=experiment_reports/agentic/appworld_probe_transfer/predictions
 --expected-count tau2_retail=114 --expected-count swe_bench=500 --expected-count appworld_test_normal=168
 --benchmark-summary tau2_retail=experiment_reports/agentic/tau2_retail_full/metrics/summary.json
 --benchmark-summary swe_bench=experiment_reports/agentic/swe_bench_full/metrics/summary.json
)
.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py "${args[@]}" --layers 12 --output experiment_reports/agentic/probe_transfer
.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py "${args[@]}" --layers 4 8 12 16 20 --output experiment_reports/agentic/probe_layer_comparison
