# Matched SWE-bench utility comparison

## Motivation and research question

Does role-probe training change agentic task utility relative to the base model under the same Inspect harness? The target is a within-harness comparison, not reproduction of a model-card score.

## Evaluation setup

All six models receive the same 500 SWE-bench Verified tasks with one attempt per task. We use the stock agent, tools and scorer from pinned Inspect Evals, with high reasoning effort, BF16 weights and native context. The default 30-message and 210-second tool limits are shared across models. A cluster sandbox adapter activates repository Python, supplies localhost resolution and snapshots initial files before model edits.

Task success is successes divided by all 500 assigned tasks; evaluation errors remain in the denominator. Change from base is the difference in percentage points. Layer-12 role accuracy is the fraction of scored content tokens classified correctly by each model’s independently fitted probe. Histories are replayed whole; over-context histories retain task outcomes but have no role score.

Residual test-environment and offline-network limitations can depress scores. They are held fixed across models; absolute benchmark parity is not claimed. No task-specific fixture, test-name remapping or custom scoring repair is used in this cohort.

## Observations

| Model | Task success | Change from base | Evaluation errors | Role accuracy | Role coverage |
|---|---:|---:|---:|---:|---:|
| Base | Pending | — | — | Pending | — |
| AppWorld-trained | Pending | — | — | Pending | — |
| ASTRA-trained | Pending | — | — | Pending | — |
| DepthBench-trained | Pending | — | — | Pending | — |
| AppWorld + agentic KL (800) | Pending | — | — | Pending | — |
| AppWorld + same-batch KL | Pending | — | — | Pending | — |

Only this matched cohort enters the comparison. Earlier baseline diagnostics and previous SWE cohorts remain separate.

## Interpretation

Collection is in progress. No conclusion about the matched utility change is available yet.

[Role breakdowns and coverage](APPENDIX.md). [Reproduction and configuration](RUNNING.md).
