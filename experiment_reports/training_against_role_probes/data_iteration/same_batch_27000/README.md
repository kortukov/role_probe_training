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
as unscorable for roles. SWE is handled by the separate evaluation effort.

## Factual observations

Collection is running; training and evaluation are queued behind validation.
No new-model benchmark results are available yet.

## Message

The effect of increasing data is not yet measured. Because training remains
one epoch, the comparison will also include increased training compute.

[Reproduction](RUNNING.md)
