# tau2_retail: final-checkpoint evaluation

Task success includes all assigned attempts. Role accuracy is content-token-weighted argmax accuracy from a freshly fitted layer-12 probe, replaying whole histories.

| Task metric | Successes / assigned | Success rate |
| --- | ---: | ---: |
| retail_scorer | 66/114 | 57.89% |

Role accuracy: **35.93%**. Scored histories: **112/114**.

Histories above native context are excluded only from role metrics; task denominators retain every attempt. No histories are chunked or truncated.

These measurements do not alone establish improvement over a baseline. SWE uses the recorded Inspect generous protocol, not a model-card reproduction.

[Detailed counts and provenance](tau2_retail.json).

[Reproduction](../../RUNNING.md)
