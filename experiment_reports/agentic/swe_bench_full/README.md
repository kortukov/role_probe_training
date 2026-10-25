# GPT-OSS-20B on SWE-bench Verified

## Motivation & RQ

How well does unchanged GPT-OSS-20B solve software-repair tasks under the matched
setup used to evaluate models trained against role probes?

## Evaluation setup

We evaluate all 500 SWE-bench Verified tasks with one attempt per task, using
Inspect's agent and upstream patch scorer. Generation uses BF16 weights, high
reasoning, the native 131,072-token context limit and a 200-message budget.
Task success divides successful patches by all assigned tasks, including errors.

## Factual observations

| Solved / assigned | Task success | Evaluation errors |
|---|---:|---:|
| 181/500 | 36.2% | 7/500 |

All 500 reconstructed histories were also scored whole by the existing role
probes. Layer-12 overall role accuracy is 36.2%; this is a separate token-level
metric and only coincidentally rounds to the same percentage as task success.
[Role results and percentage confusion matrices](../probe_transfer/README.md).

## Message

This complete native-context run supersedes the earlier constrained SWE cohorts.
Seven sandbox runtime errors remain counted as task failures. This setup does not
reproduce the model card's agent/harness or its validated task subset. See the
[current four-model comparison and limitations](../../training_against_role_probes/README.md).

[Reproduction instructions](RUNNING.md)
