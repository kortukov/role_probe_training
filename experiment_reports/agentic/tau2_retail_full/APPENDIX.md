# Supporting benchmark results

## Scoring and uncertainty

The main report uses all selected tasks as the denominator, including errors as failures. The scored-only rate below excludes attempts with evaluation errors; it is not the full-task rate. The Wilson interval describes uncertainty in that scored-only proportion, not uncertainty caused by a different evaluation setup.

| Solved / scored | Scored-only success | 95% Wilson interval | All selected tasks |
|---|---:|---|---:|
| 73 / 114 | 64.04% | 54.90%–72.25% | 114 |

## Execution diagnostics

A limited attempt reaches a configured execution bound, such as time or conversation length. An output-cap hit means a model response reaches its generation-token limit. Model-call errors count individual failed calls, which can be retried; they are not the number of failed tasks. These categories can overlap.

| Diagnostic | Count |
|---|---:|
| Attempts ending in evaluation errors | 0 |
| Attempts reaching an execution limit | 4 |
| Attempts with at least one output-cap hit | 64 |
| Output-cap hits | 113 |
| Failed agent model calls | 32 |
| Recorded model retries | 63 |

No top-level task errors were recorded, despite individual model-call failures. Limit events nested inside the interaction are counted even when the top-level sample lacks a limit flag.

## Role-probe confusion matrices

See [layer 12](../probe_transfer/APPENDIX.md#tau2_retail-layer-12) and the [five-layer comparison](../probe_layer_comparison/APPENDIX.md#tau2_retail-layer-4). These measure role recognition, separately from benchmark task success.

[Complete benchmark metrics](metrics/summary.json) and [reproduction settings](RUNNING.md).
