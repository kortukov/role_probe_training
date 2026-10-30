# SWE-bench meeting snapshot

Updated: 2026-10-30T03:02:18.700138+00:00. Meeting: 30 October 2026, 15:00 Europe/London.

**Provisional saved-log snapshot; final validated 500-task scores and whole-history probe metrics remain pending.**

Completed tasks are selected by finish time and can be systematically easier or shorter. Shared-task subsets reduce task mismatch but remain completion-biased; do not interpret these as final benchmark scores.

ASTRA shards 0–1 use concurrency 8 and shards 2–3 use 16; other models use 4. Per-attempt budgets, reasoning effort and scorer are unchanged. Batching and contention can affect sampling and timing.

| Model | Saved / assigned | Successes among saved | Saved errors | Final score |
|---|---:|---:|---:|---|
| Base | 209/500 | 108 | 1 | Pending validation |
| AppWorld-trained | 0/500 | 0 | 0 | Pending validation |
| ASTRA-trained | 0/500 | 0 | 0 | Pending validation |
| DepthBench-trained | 0/500 | 0 | 0 | Pending validation |
| AppWorld + agentic KL (800) | 0/500 | 0 | 0 | Pending validation |
| AppWorld + same-batch KL | 0/500 | 0 | 0 | Pending validation |
| AppWorld + same-batch KL (27,000) | 0/500 | 0 | 0 | Pending validation |

All-seven shared completed task count: 0.

| Trained model vs baseline | Shared completed tasks | Base successes | Trained successes |
|---|---:|---:|---:|
| AppWorld-trained | 0 | 0 | 0 |
| ASTRA-trained | 0 | 0 | 0 |
| DepthBench-trained | 0 | 0 | 0 |
| AppWorld + agentic KL (800) | 0 | 0 | 0 |
| AppWorld + same-batch KL | 0 | 0 | 0 |
| AppWorld + same-batch KL (27,000) | 0 | 0 | 0 |

Each row above uses its own shared task IDs; rows with different subsets are not comparable to each other. Errors are retained in subset denominators. The final evaluation denominator remains 500 per model.

Whole-history role metrics will follow validated task collection. No attempt is regenerated to improve the score.
