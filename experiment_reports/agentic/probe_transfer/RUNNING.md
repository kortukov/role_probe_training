# Reproduction

Predictions use schema v1, one gzip JSON per trajectory. Each contains original messages, normalized replay messages, token IDs/offsets, labels, score mask, context boundaries and per-layer probabilities and argmax predictions. Input/probe hashes and job metadata are in each prediction directory’s manifest.

Generate the report and standalone viewer from the repository root:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py --dataset 'dolci_tool_use=experiment_reports/agentic/probe_transfer/predictions/dolci' --dataset 'tau2_retail=experiment_reports/agentic/probe_transfer/predictions/retail' --dataset 'swe_bench=experiment_reports/agentic/probe_transfer/predictions/swe_selected' --expected-count 'dolci_tool_use=1000' --expected-count 'tau2_retail=114' --expected-count 'swe_bench=500' --benchmark-summary 'tau2_retail=experiment_reports/agentic/tau2_retail_full/metrics/summary.json' --benchmark-summary 'swe_bench=experiment_reports/agentic/swe_bench_full/metrics/summary.json' --layers 12 --output experiment_reports/agentic/probe_transfer
```

Add another `--dataset NAME=PATH` to include a dataset. See `src/experiments/agentic_probes/run.py --help` for inference and `src/experiments/agentic_eval/dolci.py` for seeded sampling. Generated predictions, trajectories and HTML remain on cluster storage; code, manifests and aggregate metrics are versioned.

## Inputs and validation

The [prediction manifests](metrics/prediction_manifests.json) record source/probe hashes, hook sites, role order, context policy and Slurm jobs. [SWE selection](metrics/swe_prediction_selection.json) maps exactly 500 canonical tasks to immutable prediction files; repair selection validates messages, tools, scores, errors and source logs. Original unselected attempts remain under ignored `runs/` and `predictions/` directories.

Canonical chat inputs:

- `experiment_reports/agentic/dolci_1000/data/trajectories.jsonl`
- `experiment_reports/agentic/tau2_retail_full/data/trajectories.jsonl`
- `experiment_reports/agentic/swe_bench_full/data/trajectories.jsonl`

Both reports/viewers can be rebuilt with `bash src/experiments/agentic_probes/final_reports.sh`. This refreshes metrics, appendix tables and viewers while preserving authored README and RUNNING prose. Confusion matrices are independently recomputed from saved token predictions during report generation. Expected dataset counts and all requested layers are checked. [Slurm timings](metrics/slurm_jobs.txt) record measured costs.

```bash
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/playwright" .tools/bin/uv run --isolated --no-project --with playwright python src/experiments/agentic_probes/viewer_check.py experiment_reports/agentic/probe_transfer/viewer.html
```

The HTML is a single downloadable file; no server or adjacent files are needed. Use a modern browser supporting `DecompressionStream`. Per-token JSON files retain all five layers even when this viewer displays only layer 12.

Both complete HTML files passed offline Chromium checks. Their byte sizes, SHA256 values and coverage checks are recorded in [validation.json](metrics/validation.json).

## Recorded execution settings

GPT-OSS-20B, revision `6cee5e81ee83917806bbde320786a8fb61efebee`; zero-based post-attention-layernorm / pre-MLP probes. Replay uses the checkpoint dequantized to BF16 in Transformers, matching probe fitting; benchmark generation uses vLLM MXFP4. Ground truth comes from structured roles and analysis channels. Accuracy is argmax correctness, not mean true-role probability. Structural tokens and synthetic tool-schema context are excluded from scores. All content tokens are retained; long traces use explicitly recorded 16,384-token context windows with position/context resets. Shorter trajectories preserve causal context throughout. This is standardized retrospective Harmony replay, not a byte-identical reconstruction of serving prompts. Dolci has no analysis-channel labels. Probe fitting warnings in the baseline remain applicable.

## Viewer artifact

The layer-12 HTML is 470 MB (decimal); the five-layer HTML is 1.73 GB. Both are excluded from Git and remain on cluster storage. Offline browser checks covered all datasets, each available layer and the longest history per dataset.
