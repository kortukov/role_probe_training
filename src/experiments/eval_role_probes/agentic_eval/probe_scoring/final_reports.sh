#!/bin/bash
# Refresh both metrics/appendices/viewers; preserve authored README and RUNNING files.
set -euo pipefail
args=(
 --dataset tau2_retail=experiment_reports/eval_role_probes/agentic_eval/probe_scoring/probe_transfer/predictions/retail
 --dataset swe_bench=experiment_reports/eval_role_probes/agentic_eval/probe_scoring/probe_transfer/predictions/swe
 --dataset appworld_test_normal=experiment_reports/eval_role_probes/agentic_eval/appworld/probe_transfer/predictions
 --expected-count tau2_retail=114 --expected-count swe_bench=500 --expected-count appworld_test_normal=168
 --benchmark-summary tau2_retail=experiment_reports/eval_role_probes/agentic_eval/tau2_retail/base_trajectories/metrics/summary.json
 --benchmark-summary swe_bench=experiment_reports/eval_role_probes/agentic_eval/swe_bench/base_trajectories/metrics/summary.json
)
.tools/bin/uv run --frozen --no-sync python src/experiments/eval_role_probes/agentic_eval/probe_scoring/report.py "${args[@]}" --layers 12 --output experiment_reports/eval_role_probes/agentic_eval/probe_scoring/probe_transfer
.tools/bin/uv run --frozen --no-sync python src/experiments/eval_role_probes/agentic_eval/probe_scoring/report.py "${args[@]}" --layers 4 8 12 16 20 --output experiment_reports/eval_role_probes/agentic_eval/probe_scoring/probe_layer_comparison
