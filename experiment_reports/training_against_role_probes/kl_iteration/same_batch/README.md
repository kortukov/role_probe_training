# AppWorld KL variant: same_batch

## Motivation & RQ

Does changing KL data preserve task success while improving role recovery?

## Evaluation setup

[Shared setup and reproduction](../RUNNING.md). Task success includes all assigned attempts. Overall role accuracy is token-weighted accuracy on whole histories within native context, using a fresh standard-data layer-12 probe.

## Factual observations

All values are percentages.

| Benchmark | Task success | Overall role | Histories scored |
|---|---:|---:|---:|
| AppWorld test_normal | 36.9 | 99.3 | 168/168 |
| τ² retail | 54.4 | 78.8 | 114/114 |

[Confusions and coverage](APPENDIX.md).

## Message

Task success exceeds the original AppWorld-trained model on 2/2 evaluated benchmarks, while matching or exceeding base task success on 1/2. Role accuracy and task success must be assessed together; per-role gains are not uniform. SWE-bench is deferred pending fixes by another agent. See the [main comparison](../../README.md).

[Reproduction](../RUNNING.md)
