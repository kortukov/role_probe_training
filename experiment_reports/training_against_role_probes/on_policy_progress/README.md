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

## Runs

Role accuracy is for the ten roles at layer 16, overall / balanced over the roles present, in %.

| Run | Rollouts | Regenerations | Updates | AppWorld task | AppWorld role | Retail task | Retail role | State |
|---|---|---|---|---|---|---|---|---|
| Base model | – | – | 0 | 21.4 | pending | pending | pending | reference; ten-role scoring queued |
| `small_k1` | 540 | 1 (off-policy) | 34 | | | | | submitted 2026-11-13 |
| `small_k2` | 540 | 2 | 34 | | | | | submitted 2026-11-13 |
| `small_k6` | 540 | 6 | 36 | | | | | submitted 2026-11-13 |
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
