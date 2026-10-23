# Reproduction

## Input and scoring

Input: `experiment_reports/agentic/appworld/data/test_normal.jsonl` (168 tasks).
The existing probes are frozen; no probe or model training uses this test split.
Layer 12 and 16 weights come from `experiment_reports/role_probes/gptoss20b_baseline/probes/`;
layers 4, 8, 20 from `experiment_reports/role_probes/gptoss20b_layers/probes/`.

```bash
sbatch --exclude=vca-gpu-0701-03 src/experiments/agentic_probes/appworld.sbatch
sbatch --dependency=afterok:SCORING_ARRAY --kill-on-invalid-dep=yes src/experiments/agentic_probes/appworld_report.sbatch
```

Four one-GPU shards read all five zero-based post-attention-layernorm / pre-MLP
probes in the same forward. Scoring uses GPT-OSS-20B dequantized to BF16 and the
existing eager-attention implementation. Independent 16384-token contexts retain
all tokens but restart positions and lose earlier context in later segments.
Retrospective Harmony preserves reasoning, tool names/calls and message roles;
it does not reproduce the original serving prompts exactly. Role labels come
from metadata. Headers, special tokens and synthetic tool-schema context are
excluded from accuracy. The protocol matches the existing agentic probe reports.

The four `predictions/shard*/manifest.json` files record input/probe hashes and
configuration. Per-trajectory compressed JSON stores probabilities, predictions,
labels, offsets, messages and score masks. Large predictions and the data-filled
viewer remain excluded from Git. Scoring array: 1492721; dependent report job: 1492726.

## Report and standalone viewer

The CPU report job checks coverage and confusion matrices, writes `metrics.json`
and `APPENDIX.md`, and creates `viewer.html` with all data embedded. Download the
HTML and open it in a modern browser supporting DecompressionStream. No adjacent
files or server are needed. Add further datasets using repeated `--dataset` and
`--expected-count` arguments to `src/experiments/agentic_probes/report.py`.

```bash
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/playwright" .tools/bin/uv run --isolated --no-project --with playwright python src/experiments/agentic_probes/viewer_check.py experiment_reports/agentic/appworld_probe_transfer/viewer.html
```
