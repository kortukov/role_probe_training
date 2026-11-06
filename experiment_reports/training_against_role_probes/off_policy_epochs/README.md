# Off-policy training with one rollout per task per epoch

## Motivation & RQ

How do task competence and role perception change over ten training epochs when
we refresh AppWorld rollouts at the start of each epoch?
The [paired experiment](../on_policy_epochs/README.md)
changes only the rollout refresh schedule.

## Evaluation setup

**Training.** Start from GPT-OSS-20B and train for ten epochs over the same 90
AppWorld training tasks, using one full rollout per task per epoch. Each epoch
has six updates (five batches of 16 tasks and one of 10), so each run generates
900 rollouts and performs 60 updates. Task order and rollout seeds match across
runs. All 90 histories are generated with the epoch-start policy before any update. That fixed pool is consumed once, then replaced next epoch; later batches therefore use an older policy.
No attempts are selected by success. The loss encourages correct five-role
classification at layer 12, with exact KL(current ‖ base) on generated assistant
tokens and the same optimizer settings as the earlier batch-16 experiments.
The training probe is refitted after each update on the original 250 standard
C4/Dolma passages, using all content tokens and up to 1,024 tokens per passage.

**Evaluation.** Evaluate the initial checkpoint (epoch 0) and every epoch-end
checkpoint on all 114 τ² retail tasks, one fresh attempt per task, with matched
BF16 generation settings and a GPT-4.1 customer simulator. Retail tasks never
enter training. Fit a fresh layer-12 linear probe for each checkpoint on the
same standard 250 passages and original passage split. Probes read
post-attention-layernorm outputs / pre-MLP inputs. Roles come from message and
reasoning-channel metadata; headers and synthetic tool schemas are excluded.
Each checkpoint is evaluated on its own conversations.

**Metrics and baseline.** Task success is successes divided by all 114 assigned
tasks, with evaluation errors retained as failures. Overall role accuracy is
content-token-weighted argmax accuracy. Per-role accuracy is recall among
content tokens with that true role. These measure role recovery separately
from task competence. Replay every scored conversation whole, without chunking
or truncation; histories above native context are excluded only from role
metrics. Coverage is reported with the appendix matrices. Epoch 0 provides a
matched untrained baseline; the older 64.0% retail result used different base
precision and is not substituted for this baseline.

## Factual observations

<!-- epoch-results -->
| Epoch | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 52.6 | 28.6 | 6.6 | 31.4 | 17.6 | 86.9 | 1.6 |
| 1 | 49.1 | 33.6 | 8.1 | 38.0 | 20.8 | 87.7 | 3.7 |
| 2 | 58.8 | 38.3 | 20.5 | 47.6 | 32.0 | 73.9 | 21.9 |
| 3 | 48.2 | 49.5 | 33.5 | 46.3 | 49.8 | 70.7 | 26.1 |
| 4 | 58.8 | 46.0 | 41.2 | 34.6 | 44.2 | 67.6 | 13.3 |
| 5 | 61.4 | 53.9 | 41.1 | 30.3 | 61.6 | 55.8 | 8.2 |
| 6 | 55.3 | 55.1 | 41.8 | 25.8 | 60.6 | 60.9 | 5.1 |
| 7 | 53.5 | 53.9 | 41.2 | 27.1 | 58.9 | 60.5 | 2.9 |
| 8 | 55.3 | 51.1 | 41.5 | 23.4 | 54.4 | 63.2 | 1.7 |

All values are percentages. Epoch 0 is the untrained checkpoint evaluated with the same protocol. [Checkpoint coverage and confusion matrices](APPENDIX.md).
<!-- /epoch-results -->

## Message

<!-- epoch-message -->
The latest evaluated checkpoint is epoch 8: task success is 55.3% and overall role accuracy is 51.1%. Relative to epoch 0, these change by +2.6 and +22.5 percentage points, respectively. The table gives the measured progression across completed epochs. These single-attempt measurements are descriptive. Role accuracy and task success measure different behavior; differing role-score coverage can affect comparisons. Compare both refresh schedules at matched epochs before attributing differences to refresh frequency.
<!-- /epoch-message -->

[Reproduction and execution status](RUNNING.md)
