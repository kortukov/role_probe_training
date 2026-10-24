# GPT-OSS-20B on full SWE-bench Verified

## Motivation & RQ

How often does GPT-OSS-20B solve software-repair tasks in the Inspect evaluation setup, and what role-probe behavior appears in its tool-using histories?

## Evaluation setup

**Tasks and agent.** We evaluate all 500 SWE-bench Verified tasks. GPT-OSS-20B inspects and edits a repository using Inspect's standard agent tools; the benchmark test suite scores the resulting patch. We retain one model attempt per task, including failed and interrupted attempts. Infrastructure failures before the model starts may be retried.

**Metrics.** Task success divides solved tasks by all 500 tasks, counting evaluation errors as failures. We separately report attempts that could not be scored because the model API failed. This distinction matters because a scored-only success rate would exclude those failures.

## Factual observations

| Tasks solved | Task success | Attempts ending in model API errors |
|---|---:|---:|
| 11 / 500 | 2.20% | 226 / 500 |

All 500 selected histories contain tool replies. Execution limits affected 271 attempts, and malformed model output caused many API failures. The [appendix](APPENDIX.md) gives scored-only results and further diagnostics; the [probe report](../probe_transfer/README.md) evaluates the saved conversational tokens.

## Message

This setup collects agentic traces but fails to reproduce the published 60.7% SWE-bench score. The agent tools, evaluation subset and generation settings differ from the published setup, and the large number of API failures prevents treating this result as a reliable estimate of the model's software-engineering capability. [OpenAI model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations).

The histories remain useful for studying role recognition in actual tool interactions, but represent many unsuccessful or interrupted attempts. The repaired setup passes targeted request checks and completes one task through scoring without infrastructure errors ([diagnosis](APPENDIX.md#serving-diagnosis)). That patch did not solve the issue. Two repaired shards completed (42/250 solved), but 204/250 attempts encountered the 32K serving context limit; two other shards failed before inference. This partial result is diagnostic only. See the [current failure audit](../../training_against_role_probes/APPENDIX.md#swe-bench-failure-audit). A complete repaired capability estimate remains unavailable.

See [RUNNING.md](RUNNING.md) for configuration, infrastructure repairs, source artifacts and viewer reproduction.
