# Increasing AppWorld trajectories per task

## Motivation & RQ

Does increasing AppWorld training data from 30 to 300 trajectories per task
improve task performance and role perception with same-batch KL?

## Evaluation setup

The training set covers the same 90 tasks, expanding from 2,700 to 27,000
base-model trajectories without outcome filtering. The new model starts from
the base model with the previous same-batch KL objective and optimizer settings.
One epoch increases both data exposure and updates, from 169 to 1,688.

Evaluation uses AppWorld normal test (168 tasks) and retail (114 tasks).
Task success counts all assigned attempts. Role accuracy is token-weighted
accuracy from a fresh standard-data layer-12 probe on complete histories;
histories above native context remain in task denominators and are reported
as unscorable for roles. Per-role columns report recall: the fraction of content
tokens of that true role classified correctly. The comparison baseline is the
previous 2,700-trajectory same-batch KL model on these same evaluation tasks.
SWE is handled by the separate evaluation effort.

## Factual observations

All values are percentages. All 168 AppWorld and 114 retail histories were
scorable, with no evaluation errors. The 27,000-trajectory model solved 50/168
AppWorld tasks and 62/114 retail tasks.

| Benchmark | Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AppWorld test_normal | AppWorld + same-batch KL (27,000) | 29.8 | 99.8 | 100.0 | 100.0 | 99.8 | 99.4 | 100.0 |
| τ² retail | AppWorld + same-batch KL (27,000) | 54.4 | 77.9 | 78.5 | 30.2 | 73.5 | 91.8 | 93.7 |

[Confusions and coverage](APPENDIX.md).

## Message

Relative to the 2,700-trajectory same-batch KL model, AppWorld test_normal task success changes by -7.1 percentage points and overall role accuracy by +0.5 points. τ² retail: task success changes by +0.0 percentage points and overall role accuracy by -0.8 points.

Increasing the data and updates did not improve observed task success: AppWorld
success decreased and retail success was unchanged.

The data-size comparison holds the same-batch KL objective and optimizer settings fixed, but one epoch on ten times more data also increases updates from 169 to 1,688. It does not isolate data diversity from training compute. Each model is evaluated on its own histories, and overall role accuracy need not imply uniform improvement across roles.

[Reproduction](RUNNING.md)
