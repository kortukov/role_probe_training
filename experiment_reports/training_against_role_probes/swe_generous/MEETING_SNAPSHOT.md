# SWE-bench meeting snapshot

Updated: 2026-11-01T09:34:19.459570+00:00. Meeting: 30 October 2026, 15:00 Europe/London.

**All seven 500-task scores and all seven whole-history role results are validated.**

All models now cover the same 500 tasks. Earlier partial snapshots were completion-biased; the complete scores below retain every attempt, including errors.

ASTRA shards 0–1 use concurrency 8 and shards 2–3 use 16; other models use 4. Per-attempt budgets, reasoning effort and scorer are unchanged. Batching and contention can affect sampling and timing.

| Model | Saved / assigned | Successes among saved | Saved errors | Final score |
|---|---:|---:|---:|---|
| Base | 500/500 | 254 | 2 | 50.8% (validated) |
| AppWorld-trained | 500/500 | 22 | 0 | 4.4% (validated) |
| ASTRA-trained | 500/500 | 133 | 1 | 26.6% (validated) |
| DepthBench-trained | 500/500 | 71 | 0 | 14.2% (validated) |
| AppWorld + agentic KL (800) | 500/500 | 241 | 0 | 48.2% (validated) |
| AppWorld + same-batch KL | 500/500 | 243 | 0 | 48.6% (validated) |
| AppWorld + same-batch KL (27,000) | 500/500 | 257 | 1 | 51.4% (validated) |

All-seven shared completed task count: 500.

| Trained model vs baseline | Shared completed tasks | Base successes | Trained successes |
|---|---:|---:|---:|
| AppWorld-trained | 500 | 254 | 22 |
| ASTRA-trained | 500 | 254 | 133 |
| DepthBench-trained | 500 | 254 | 71 |
| AppWorld + agentic KL (800) | 500 | 254 | 241 |
| AppWorld + same-batch KL | 500 | 254 | 243 |
| AppWorld + same-batch KL (27,000) | 500 | 254 | 257 |

Each row above uses its own shared task IDs; rows with different subsets are not comparable to each other. Errors are retained in subset denominators. The final evaluation denominator remains 500 per model.

Whole-history role metrics are available for all seven models. No attempt is regenerated to improve the score.

AppWorld collection and assembly are complete: 22/500 successes (4.4%), with zero sample or export errors and no message-limit or context-limit stops. [Validated task provenance](appworld_task_results.json).

Its layer-12 role probe correctly classifies 87.55% of scored content tokens (token-weighted accuracy); the mean per-conversation accuracy is 87.08%. Coverage is 500/500 whole conversations, replayed without chunking or truncation, with no exclusions. Roles are system, user, reasoning, assistant, and tool. These role metrics do not measure task success. [Role metrics and confusion counts](appworld_role_results.json).

DepthBench task evaluation is complete and validated: 71/500 successes (14.2%), with zero sample or export errors. Two attempts encountered model context-limit stops and remain in the denominator. Its whole-history probe scoring is complete. [Validated task provenance](depthbench_task_results.json).

DepthBench layer-12 role accuracy is 48.36% of scored content tokens (token-weighted); mean per-conversation accuracy is 50.22%. Coverage is 498/500 whole conversations, with two histories excluded because they exceed the native context limit. No histories were chunked or truncated; both excluded histories remain in the 500-task success denominator. [Role metrics and confusion counts](depthbench_role_results.json).

Baseline task evaluation is complete and validated: 254/500 successes (50.8%). Two scorer timeouts remain in the denominator as failures. All original attempts are retained. Whole-history role scoring is complete. [Validated task provenance](base_task_results.json).

Agentic-KL task collection is complete: 241/500 successes (48.2%), 2.6 percentage points below the matched baseline. All four completed shard exports passed expected-ID, hash, and settings validation, with zero sample/export errors. Two attempts recorded limit events and eight recorded model context stops; all remain in the denominator. The assembled output has also passed hash validation; role scoring is complete. [Validated task provenance](agentickl_task_results.json).

Same-batch KL (2,700 training trajectories) task evaluation is complete and validated: 243/500 successes (48.6%), 2.2 percentage points below baseline, with zero sample/export errors. Thirteen attempts recorded model context stops; all remain in the denominator. Role scoring is complete. [Validated task provenance](samebatch_task_results.json).

Newly validated layer-12 token-weighted role accuracy: baseline 41.17% (481/500 histories), agentic KL 94.29% (493/500), and same-batch KL 93.92% (485/500). Over-context exclusions affect only role metrics; all 500 attempts remain in each task denominator. These metrics describe each model’s own eligible histories.

ASTRA task evaluation is validated: 133/500 (26.6%). One scorer timeout and four model context stops remain in the denominator. All original attempts survived the final shard’s Slurm OOM and passed saved-output integrity validation; no model or scorer reruns. [Task result](astra_task_results.json), [recovery evidence](astra_oom_saved_validation.json).

ASTRA layer-12 token-weighted role accuracy is 60.34%; mean per-conversation accuracy is 63.39%. Coverage is 498/500 whole histories, with two over-context exclusions affecting only role metrics. [Role metrics and provenance](astra_role_results.json).

The 27,000-trajectory model is validated at 257/500 (51.4%), with one scorer timeout and 12 model context stops retained. This is three more successes than baseline, a descriptive difference. [Task provenance](samebatch27000_task_results.json).

The 27,000-trajectory model’s layer-12 token-weighted role accuracy is 97.66%; mean per-conversation accuracy is 97.78%. Coverage is 485/500 whole histories; 15 over-context exclusions affect only role metrics. [Role metrics and provenance](samebatch27000_role_results.json).
