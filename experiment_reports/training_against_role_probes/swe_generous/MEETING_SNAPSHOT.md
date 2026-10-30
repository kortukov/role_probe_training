# SWE-bench meeting snapshot

Updated: 2026-10-30T14:37:38.764840+00:00. Meeting: 30 October 2026, 15:00 Europe/London.

**Provisional saved-log snapshot; the complete seven-model comparison remains pending; four complete task scores are validated.**

Completed tasks are selected by finish time and can be systematically easier or shorter. Shared-task subsets reduce task mismatch but remain completion-biased; do not interpret these as final benchmark scores.

ASTRA shards 0–1 use concurrency 8 and shards 2–3 use 16; other models use 4. Per-attempt budgets, reasoning effort and scorer are unchanged. Batching and contention can affect sampling and timing.

| Model | Saved / assigned | Successes among saved | Saved errors | Final score |
|---|---:|---:|---:|---|
| Base | 500/500 | 254 | 2 | 50.8% (validated) |
| AppWorld-trained | 500/500 | 22 | 0 | 4.4% (validated) |
| ASTRA-trained | 0/500 | 0 | 0 | Pending validation |
| DepthBench-trained | 500/500 | 71 | 0 | 14.2% (validated) |
| AppWorld + agentic KL (800) | 500/500 | 241 | 0 | 48.2% (validated) |
| AppWorld + same-batch KL | 467/500 | 229 | 0 | Pending validation |
| AppWorld + same-batch KL (27,000) | 0/500 | 0 | 0 | Pending validation |

All-seven shared completed task count: 0.

| Trained model vs baseline | Shared completed tasks | Base successes | Trained successes |
|---|---:|---:|---:|
| AppWorld-trained | 500 | 254 | 22 |
| ASTRA-trained | 0 | 0 | 0 |
| DepthBench-trained | 500 | 254 | 71 |
| AppWorld + agentic KL (800) | 500 | 254 | 241 |
| AppWorld + same-batch KL | 467 | 241 | 229 |
| AppWorld + same-batch KL (27,000) | 0 | 0 | 0 |

Each row above uses its own shared task IDs; rows with different subsets are not comparable to each other. Errors are retained in subset denominators. The final evaluation denominator remains 500 per model.

Whole-history role metrics are available for AppWorld and DepthBench; the remaining models are pending. No attempt is regenerated to improve the score.

AppWorld collection and assembly are complete: 22/500 successes (4.4%), with zero sample or export errors and no message-limit or context-limit stops. [Validated task provenance](appworld_task_results.json).

Its layer-12 role probe correctly classifies 87.55% of scored content tokens (token-weighted accuracy); the mean per-conversation accuracy is 87.08%. Coverage is 500/500 whole conversations, replayed without chunking or truncation, with no exclusions. Roles are system, user, reasoning, assistant, and tool. These role metrics do not measure task success. [Role metrics and confusion counts](appworld_role_results.json).

DepthBench task evaluation is complete and validated: 71/500 successes (14.2%), with zero sample or export errors. Two attempts encountered model context-limit stops and remain in the denominator. Its whole-history probe scoring is complete. [Validated task provenance](depthbench_task_results.json).

DepthBench layer-12 role accuracy is 48.36% of scored content tokens (token-weighted); mean per-conversation accuracy is 50.22%. Coverage is 498/500 whole conversations, with two histories excluded because they exceed the native context limit. No histories were chunked or truncated; both excluded histories remain in the 500-task success denominator. [Role metrics and confusion counts](depthbench_role_results.json).

Baseline task evaluation is complete and validated: 254/500 successes (50.8%). Two scorer timeouts remain in the denominator as failures. All original attempts are retained. Whole-history role scoring is pending. [Validated task provenance](base_task_results.json).

Agentic-KL task collection is complete: 241/500 successes (48.2%), 2.6 percentage points below the matched baseline. All four completed shard exports passed expected-ID, hash, and settings validation, with zero sample/export errors. Two attempts recorded limit events and eight recorded model context stops; all remain in the denominator. The assembled output has also passed hash validation; role scoring remains queued. [Validated task provenance](agentickl_task_results.json).
