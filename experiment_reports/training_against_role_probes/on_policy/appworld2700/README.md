# AppWorld on-policy training: 2,700 episodes, batch 16

## Motivation & RQ

Does updating from freshly generated AppWorld episodes preserve useful task
behavior while improving role recovery? This variant uses 2,700 episodes
and effective batch 16; the [parent comparison](../../README.md) places it
beside the other episode and batch budgets.

## Evaluation setup

Training starts from GPT-OSS-20B on the same 90 AppWorld training tasks, with
30 attempts per task. Each update combines role-probe cross-entropy
and reverse KL to the base at generated assistant-token contexts, then refits a
linear layer-12 probe on the standard 250 neutral passages. Evaluation tasks are
held out from training. The probe reads post-attention-layernorm pre-MLP inputs.

Fresh evaluation uses AppWorld normal test (168 tasks), τ² retail (114 tasks),
and SWE-bench Verified (500 tasks), with one attempt per task. Task success uses
all assigned attempts, counting evaluation errors as failures. Overall role
accuracy is the fraction of scored content tokens assigned the correct role;
whole histories above native context have no role score but retain task outcomes.
Role coverage is scored histories divided by assigned histories. It does not
measure task competence. Models are evaluated on their own conversations.

## Factual observations

| Benchmark | Task success | Overall role accuracy | Role coverage |
|---|---:|---:|---:|
| AppWorld normal test | 26.2% (44/168) | 68.2% | 168/168 |
| τ² retail | 52.6% (60/114) | 52.5% | 113/114 |
| SWE-bench Verified | 47.4% (237/500) | 38.7% | 475/500 |

[Saved results and provenance](post_training/results/) support these values.
The [parent appendix](../../APPENDIX.md) supplies the comparison's role confusion
matrices, context coverage and training-time measurements.

## Message

The 2,700-episode run does not improve AppWorld task success or role accuracy
over 270 episodes at batch 16. SWE task success is 47.4%, compared with 44.0%
for 270 episodes at batch 16. Episode
count also changes the number of updates, so data and optimization effort are
not isolated.

[Reproduction, checkpoint paths and current job records](RUNNING.md)
