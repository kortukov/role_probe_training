# appworld_test: final-checkpoint evaluation

Task success includes all assigned attempts. Role accuracy is content-token-weighted argmax accuracy from a freshly fitted layer-12 probe, replaying whole histories.

| Task metric | Successes / assigned | Success rate |
| --- | ---: | ---: |
| appworld_success | 44/168 | 26.19% |

Role accuracy: **68.18%**. Scored histories: **168/168**.

Histories above native context are excluded only from role metrics; task denominators retain every attempt. No histories are chunked or truncated.

These measurements do not alone establish improvement over a baseline. SWE uses the recorded Inspect generous protocol, not a model-card reproduction.

[Detailed counts and provenance](appworld_test.json).

[Reproduction](../../RUNNING.md)
