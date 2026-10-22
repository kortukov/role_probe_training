# Reproduction

## Inputs and probes

The evaluation contains 114 retail and 500 SWE-bench Verified trajectories:

- `experiment_reports/agentic/tau2_retail_full/data/trajectories.jsonl`
- `experiment_reports/agentic/swe_bench_full/data/trajectories.jsonl`

Predictions are gzip JSON records under `experiment_reports/agentic/probe_transfer/predictions/retail/` and `predictions/swe/`. Each stores the original messages, replay text, token IDs/offsets, metadata-derived labels, scoring mask, context boundaries, and probabilities/predictions for all five layers.

Use the [baseline weights](../../role_probes/gptoss20b_baseline/RUNNING.md) for layers 12/16 and [additional weights](../../role_probes/gptoss20b_layers/RUNNING.md) for layers 4/8/20. [Prediction manifests](../probe_transfer/metrics/prediction_manifests.json) record the inference configuration and probe hashes; [SWE selection](../probe_transfer/metrics/swe_prediction_selection.json) identifies the 500 retained files and their hashes.

## Inference

Run `sbatch src/experiments/agentic_probes/benchmark.sbatch INPUT_JSONL OUTPUT_DIRECTORY` from the repository root. The job reads all five probes in one forward pass per context segment. It uses GPT-OSS-20B dequantized to BF16 with original eager attention; benchmark generation uses vLLM MXFP4. Hooks read zero-based `post_attention_layernorm` outputs, the pre-MLP input. All content tokens are retained; 16,384-token nonoverlapping segments reset position and context. Header/special tokens and synthetic tool-schema context are excluded from accuracy.

The renderer is standardized retrospective Harmony, not a reconstruction of exact generation-time prompts. Model revisions and precision settings are pinned in configuration and saved manifests. Saved weights retain their fitting/convergence metadata.

## Metrics and standalone viewer

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py \
  --dataset tau2_retail=experiment_reports/agentic/probe_transfer/predictions/retail \
  --dataset swe_bench=experiment_reports/agentic/probe_transfer/predictions/swe \
  --expected-count tau2_retail=114 --expected-count swe_bench=500 \
  --benchmark-summary tau2_retail=experiment_reports/agentic/tau2_retail_full/metrics/summary.json \
  --benchmark-summary swe_bench=experiment_reports/agentic/swe_bench_full/metrics/summary.json \
  --layers 4 8 12 16 20 --output experiment_reports/agentic/probe_layer_comparison
```

`bash src/experiments/agentic_probes/final_reports.sh` refreshes both experiments. Regeneration preserves authored README/RUNNING prose. It checks coverage, requested layers and confusion matrices against saved per-token predictions. Add a dataset with another `--dataset NAME=PATH` argument and its expected count.

The local `viewer.html` embeds all data and can be downloaded and opened without a server or adjacent files. A modern browser with `DecompressionStream` support is required. It supports dataset/layer/conversation selection, role coloring, mismatch filtering and token probability inspection. Data, predictions and data-filled HTML remain excluded from Git.

```bash
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/playwright" .tools/bin/uv run --isolated --no-project --with playwright python src/experiments/agentic_probes/viewer_check.py experiment_reports/agentic/probe_layer_comparison/viewer.html
```

Artifact hashes and browser/coverage checks are recorded in [validation.json](../probe_transfer/metrics/validation.json). [Slurm timings](../probe_transfer/metrics/slurm_jobs.txt) record the current runs.
