# Matched SWE-bench utility comparison

## Motivation and research question

Does role-probe training change agentic task utility relative to the base model under the same Inspect harness? The target is a within-harness comparison, not reproduction of a model-card score.

## Evaluation setup

All seven models receive the same 500 SWE-bench Verified tasks with one attempt per task. We use the stock agent, tools and scorer from pinned Inspect Evals, with high reasoning effort, BF16 weights and native context. The explicit 2,000-message and 210-second tool limits are shared across models. A cluster sandbox adapter activates repository Python, supplies localhost resolution and snapshots initial files before model edits.

Task success is successes divided by all 500 assigned tasks; evaluation errors remain in the denominator. Change from base is the difference in percentage points. Evaluation errors include scorer timeouts. Limit stops count attempts with logged evaluation limit events; model context stops are recorded separately in the task-result JSON files. Layer-12 role accuracy is the fraction of scored content tokens classified correctly by each model’s independently fitted probe. Histories are replayed whole; over-context histories retain task outcomes but have no role score.

Residual test-environment and offline-network limitations can depress scores. They are held fixed across models; absolute benchmark parity is not claimed. No task-specific fixture, test-name remapping or custom scoring repair is used in this cohort.

ASTRA uses concurrency 8 for shards 0–1 and 16 for shards 2–3; other models use 4. Per-attempt budgets and scoring are unchanged. Batching may affect numerical sampling and wall-clock contention, so this scheduling difference limits strict matching.

## Observations

| Model | Task success | Change from base | Evaluation errors | Limit stops | Role accuracy | Role coverage |
|---|---:|---:|---:|---:|---:|---:|
| Base | 50.8% | +0.0 | 2 | 1 | 41.2% | 481/500 |
| AppWorld-trained | 4.4% | -46.4 | 0 | 0 | 87.5% | 500/500 |
| ASTRA-trained | 26.6% | -24.2 | 1 | 0 | 60.3% | 498/500 |
| DepthBench-trained | 14.2% | -36.6 | 0 | 0 | 48.4% | 498/500 |
| AppWorld + agentic KL (800) | 48.2% | -2.6 | 0 | 2 | 94.3% | 493/500 |
| AppWorld + same-batch KL | 48.6% | -2.2 | 0 | 0 | 93.9% | 485/500 |
| AppWorld + same-batch KL (27,000) | 51.4% | +0.6 | 1 | 0 | Pending | — |

Only this matched cohort enters the comparison. Earlier baseline diagnostics and previous SWE cohorts remain separate.

## Interpretation

The direction and size of the utility change vary by trained model; the table reports each matched difference.
These are descriptive single-attempt comparisons, not evidence of model-card parity or statistical significance.

[Role breakdowns and coverage](APPENDIX.md). [Reproduction and configuration](RUNNING.md).
