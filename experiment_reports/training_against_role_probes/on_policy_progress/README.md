# Progress log: from off-policy to on-policy training

Working notes, not a finished report. Nothing here is in the
[training status page](../README.md) until we decide a result holds.

## Question

Off-policy training (rollouts generated once by the base model) worked under the old
templates; regenerating rollouts before every update did not. Where between the two does
training stop working? We keep the number of rollouts and updates fixed and vary only how
many times the rollouts are regenerated from the current model during training.

## Setup shared by all runs

Unified setup: Harmony renderer, ten-role probe (one role per Harmony header form) at layer
16, refitted to convergence after every update, temperature 1. GPT-OSS-20B with rank-64 LoRA, 90 AppWorld training tasks, updates of 16
rollouts, learning rate 5e-5, KL coefficient 16, seed 123. One trainer for every run
(`src/role_probe_training/epoch_training/train.py`): at the start of each epoch it generates
the epoch's rollouts with the current model, then uses each once.

Fast evaluation of the final checkpoint: AppWorld test (168 tasks) and τ² retail (114 tasks),
one attempt per task, with a probe fitted from scratch on the trained model. SWE-bench is
left for the runs worth it. One attempt per task means roughly ±7 points on AppWorld and ±9
on retail task success.

## Base model, both role sets

Unchanged GPT-OSS-20B at temperature 1, probes fitted on neutral passages, layer 16. Task
success: AppWorld 21.4%, retail 57.0%, SWE-bench 52.0%. Role accuracy in %.

Five roles (held-out passage accuracy 85.3%):

| Benchmark | Overall | System | User | Reasoning | Assistant | Tool |
|---|---|---|---|---|---|---|
| AppWorld test | 47.0 | 0.0 | 91.9 | 52.3 | 67.9 | 40.8 |
| τ² retail | 51.0 | 0.0 | 12.8 | 74.5 | 99.0 | 7.3 |
| SWE-bench | 30.8 | 0.0 | 39.4 | 42.7 | 87.7 | 13.5 |

Ten roles (held-out passage accuracy 68.4%). A dash means the role does not occur; tool
output on the analysis channel never occurs:

| Benchmark | Overall | System | Developer | User | Analysis | Commentary | Final | Tool call | Tool output | Analysis tool call |
|---|---|---|---|---|---|---|---|---|---|---|
| AppWorld test | 7.2 | 4.2 | 85.7 | 38.6 | 6.6 | – | – | 2.9 | 4.3 | 24.5 |
| τ² retail | 51.4 | 4.2 | 93.4 | 0.7 | 41.8 | – | 99.8 | 0.5 | 0.7 | 3.1 |
| SWE-bench | 5.3 | 4.2 | 79.0 | 33.9 | 2.2 | 0.0 | 98.9 | 2.5 | 0.7 | 6.5 |

Ten-role predictions merged back into the five roles, for a like-for-like comparison:

| Benchmark | Overall | System | User | Reasoning | Assistant | Tool |
|---|---|---|---|---|---|---|
| AppWorld test | 21.0 | 81.9 | 38.6 | 11.7 | 48.5 | 20.9 |
| τ² retail | 54.6 | 92.8 | 0.7 | 43.3 | 97.2 | 6.5 |
| SWE-bench | 19.1 | 78.4 | 33.9 | 9.5 | 77.1 | 15.2 |

What this shows before any training: the ten-role probe fixes the system/developer blind
spot of the five-role probe (0% to 78–93%) and reads final answers almost perfectly, but it
loses most reasoning and tool-output tokens, which are the bulk of AppWorld and SWE-bench.
On AppWorld it calls 52% of reasoning "tool output on the analysis channel" and 44% of tool
output "developer". Both probes are far better on held-out passages, where the header is
the only cue, than on agent runs.

## Runs

Role accuracy is at layer 16 in the run's own role set, overall / balanced over the roles
present, in %. Runs named `_five` use the five-role probe, the others the ten-role probe.

| Run | Rollouts | Regenerations | Updates | AppWorld task | AppWorld role | Retail task | Retail role | State |
|---|---|---|---|---|---|---|---|---|
| Base model, ten roles | – | – | 0 | 21.4 | 7.2 / 23.8 | 57.0 | 51.4 / 30.5 | reference |
| Base model, five roles | – | – | 0 | 21.4 | 47.0 / 50.6 | 57.0 | 51.0 / 38.7 | reference |
| `small_k1` | 540 | 1 (off-policy) | 34 | | | | | submitted 2026-11-13 |
| `small_k1_five` | 540 | 1 (off-policy) | 34 | | | | | submitted 2026-11-13 |
| `small_k2` | 540 | 2 | 34 | | | | | submitted 2026-11-13 |
| `small_k2_five` | 540 | 2 | 34 | | | | | submitted 2026-11-13 |
| `small_k6` | 540 | 6 | 36 | | | | | submitted 2026-11-13 |
| `small_k6_five` | 540 | 6 | 36 | | | | | submitted 2026-11-13 |
| `full_k1` | 2,700 | 1 (off-policy) | 169 | | | | | planned |
| `full_k2` | 2,700 | 2 | 170 | | | | | planned |
| `full_k5` | 2,700 | 5 | 170 | | | | | planned |
| `full_k10` | 2,700 | 10 | 170 | | | | | planned |

## Notes

- 2026-11-13. Small ladder first: 6 attempts per task in total, to see whether 34 updates
  already move role accuracy and to exercise the pipeline before the long runs. `small_k6`
  has 36 updates instead of 34 because each epoch's last batch is partial.
- 2026-11-13. Switched from five to ten roles before any run started; the first submissions
  (five roles) were cancelled in the queue.
- 2026-11-13. This is the first time the trainer and the evaluation job run in the unified
  setup; a failure in either is a pipeline finding, not a result.

[Running](RUNNING.md)
