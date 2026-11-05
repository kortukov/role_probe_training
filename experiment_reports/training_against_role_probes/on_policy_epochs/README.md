# On-policy training with one rollout per task per epoch

## Motivation & RQ

How do task competence and role perception change over ten training epochs when
we refresh AppWorld rollouts before each batch?
The [paired experiment](../off_policy_epochs/README.md)
changes only the rollout refresh schedule.

## Evaluation setup

**Training.** Start from GPT-OSS-20B and train for ten epochs over the same 90
AppWorld training tasks, using one full rollout per task per epoch. Each epoch
has six updates (five batches of 16 tasks and one of 10), so each run generates
900 rollouts and performs 60 updates. Task order and rollout seeds match across
runs. Each batch is generated from the current policy immediately before its update.
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
| 0 | 58.8 | 31.3 | 6.6 | 29.0 | 19.3 | 88.3 | 1.3 |

All values are percentages. Epoch 0 is the untrained checkpoint evaluated with the same protocol. [Checkpoint coverage and confusion matrices](APPENDIX.md).
<!-- /epoch-results -->

## Message

<!-- epoch-message -->
The latest evaluated checkpoint is epoch 0: task success is 58.8% and overall role accuracy is 31.3%. These single-attempt measurements are descriptive. Role accuracy and task success measure different behavior; differing role-score coverage can affect comparisons. Compare both refresh schedules at matched epochs before attributing differences to refresh frequency.
<!-- /epoch-message -->

[Reproduction and execution status](RUNNING.md)
