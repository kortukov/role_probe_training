# swe_bench: final-checkpoint evaluation

Task success includes all assigned attempts. Role accuracy is content-token-weighted argmax accuracy from a freshly fitted layer-12 probe, replaying whole histories.

| Task metric | Successes / assigned | Success rate |
| --- | ---: | ---: |
| swe_bench_scorer | 237.0/500 | 47.40% |

Role accuracy: **38.71%**. Scored histories: **475/500**.

Histories above native context are excluded only from role metrics; task denominators retain every attempt. No histories are chunked or truncated.

These measurements do not alone establish improvement over a baseline. SWE uses the recorded Inspect generous protocol, not a model-card reproduction.

[Detailed counts and provenance](swe_bench.json).

[Reproduction](../../RUNNING.md)
