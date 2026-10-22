# GPT-OSS-20B retail trajectory pilot

Historical pilot; the complete dataset is evaluated in the [full retail experiment](../tau2_retail_full/README.md).

## Motivation & RQ

Can a small retail evaluation produce scored, tool-using conversations for probe analysis within an hour?

## Evaluation setup

**Tasks and agent.** We selected 20 τ²-bench retail tasks. GPT-OSS-20B interacts with a GPT-4.1 customer simulator and retail tools; the benchmark checks task completion using database state and required communicated information. After an initial run frequently exhausted its generation budget, we repeated the same tasks with a larger budget. This was a pipeline correction, not a controlled comparison of budgets.

**Metrics.** Task success is the fraction of all 20 selected tasks passed, retaining failed and interrupted attempts. The 95% Wilson confidence interval expresses uncertainty in that rate. Total job time includes model startup, evaluation and trajectory export, testing the practical one-hour target.

## Factual observations

| Run | Tasks solved | Task success | 95% confidence interval |
|---|---:|---:|---|
| Initial limited-budget run | 3 / 20 | 15% | 5.2%–36.0% |
| Revised pilot | 9 / 20 | 45% | 25.8%–65.8% |

The revised pilot completed in 27 minutes and produced 20 histories containing tool replies. Two conversations reached the time limit; model-call failures and output truncation also occurred. The [appendix](APPENDIX.md) records detailed trace diagnostics. Probe results are reported on the later full collection, not as a separate evaluation of this pilot.

## Message

Inspect produced reusable agent histories within the one-hour target. The sample offers a rough task-performance signal, but its wide uncertainty interval and remaining interruptions preclude a strong capability comparison. The published retail score also concerns the original τ-bench rather than τ²-bench.

The successful collection motivated the full experiment. Keeping unsuccessful histories avoids selecting unusually easy conversations for probe evaluation.

See [RUNNING.md](RUNNING.md) for settings, artifacts and reproduction commands.
