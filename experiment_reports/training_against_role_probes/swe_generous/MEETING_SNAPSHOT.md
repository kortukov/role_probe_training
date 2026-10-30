# SWE-bench meeting snapshot

Updated: 2026-10-30T09:54:23.008585+00:00. Meeting: 30 October 2026, 15:00 Europe/London.

**Provisional saved-log snapshot; the complete seven-model comparison remains pending; validated AppWorld results are available below.**

Completed tasks are selected by finish time and can be systematically easier or shorter. Shared-task subsets reduce task mismatch but remain completion-biased; do not interpret these as final benchmark scores.

ASTRA shards 0–1 use concurrency 8 and shards 2–3 use 16; other models use 4. Per-attempt budgets, reasoning effort and scorer are unchanged. Batching and contention can affect sampling and timing.

| Model | Saved / assigned | Successes among saved | Saved errors | Final score |
|---|---:|---:|---:|---|
| Base | 418/500 | 216 | 2 | Pending validation |
| AppWorld-trained | 500/500 | 22 | 0 | 4.4% (validated) |
| ASTRA-trained | 0/500 | 0 | 0 | Pending validation |
| DepthBench-trained | 483/500 | 68 | 0 | Pending validation |
| AppWorld + agentic KL (800) | 312/500 | 162 | 0 | Pending validation |
| AppWorld + same-batch KL | 147/500 | 76 | 0 | Pending validation |
| AppWorld + same-batch KL (27,000) | 0/500 | 0 | 0 | Pending validation |

All-seven shared completed task count: 0.

| Trained model vs baseline | Shared completed tasks | Base successes | Trained successes |
|---|---:|---:|---:|
| AppWorld-trained | 418 | 216 | 19 |
| ASTRA-trained | 0 | 0 | 0 |
| DepthBench-trained | 409 | 212 | 61 |
| AppWorld + agentic KL (800) | 312 | 174 | 162 |
| AppWorld + same-batch KL | 147 | 82 | 76 |
| AppWorld + same-batch KL (27,000) | 0 | 0 | 0 |

Each row above uses its own shared task IDs; rows with different subsets are not comparable to each other. Errors are retained in subset denominators. The final evaluation denominator remains 500 per model.

Whole-history role metrics are available for AppWorld; the remaining models are pending. No attempt is regenerated to improve the score.

AppWorld collection and assembly are complete: 22/500 successes (4.4%), with zero sample or export errors and no message-limit or context-limit stops. [Validated task provenance](appworld_task_results.json).

Its layer-12 role probe correctly classifies 87.55% of scored content tokens (token-weighted accuracy); the mean per-conversation accuracy is 87.08%. Coverage is 500/500 whole conversations, replayed without chunking or truncation, with no exclusions. Roles are system, user, reasoning, assistant, and tool. These role metrics do not measure task success. [Role metrics and confusion counts](appworld_role_results.json).
