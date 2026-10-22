# Supporting benchmark results

## Scoring and uncertainty

The main report uses all selected tasks as the denominator, including errors as failures. The scored-only rate below excludes attempts with evaluation errors; it is not the full-task rate. The Wilson interval describes uncertainty in that scored-only proportion, not uncertainty caused by a different evaluation setup.

| Solved / scored | Scored-only success | 95% Wilson interval | All selected tasks |
|---|---:|---|---:|
| 11 / 274 | 4.01% | 2.26%–7.04% | 500 |

## Execution diagnostics

A limited attempt reaches a configured execution bound, such as time or conversation length. An output-cap hit means a model response reaches its generation-token limit. Model-call errors count individual failed calls, which can be retried; they are not the number of failed tasks. These categories can overlap.

| Diagnostic | Count |
|---|---:|
| Attempts ending in evaluation errors | 226 |
| Attempts reaching an execution limit | 271 |
| Attempts with at least one output-cap hit | 73 |
| Output-cap hits | 109 |
| Failed agent model calls | 774 |
| Recorded model retries | 1701 |

All remaining evaluation errors are model API retry failures. Two image failures were retried only after verifying that no model call had occurred. Two tool-name normalization errors were repaired without generating new model attempts. These selections are recorded in [provenance](metrics/provenance.json).

## Role-probe confusion matrices

See [layer 12](../probe_transfer/APPENDIX.md#swe_bench-layer-12) and the [five-layer comparison](../probe_layer_comparison/APPENDIX.md#swe_bench-layer-4). These measure role recognition, separately from benchmark task success.

[Complete benchmark metrics](metrics/summary.json) and [reproduction settings](RUNNING.md).
