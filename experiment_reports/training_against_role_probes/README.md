# Training GPT-OSS-20B to preserve conversational roles

## Motivation & RQ

Does training GPT-OSS-20B with continually refitted role probes make conversational
roles easier to recover on agent trajectories while preserving the base model's behavior?

## Evaluation setup

**Data.** We collect 30 independently seeded attempts on each of AppWorld's 90
training tasks: **2,700 trajectories**, including successful and unsuccessful
attempts. This is off-policy training: the conversations remain fixed throughout
optimization. Eight separate development tasks provide a small held-out check.

**Method.** A linear five-class probe reads layer 12's pre-MLP activations and
predicts system, user, reasoning, assistant or tool roles. Ground truth comes
from message and channel metadata; formatting tokens are excluded. Each model
update minimizes correct-role cross-entropy with the probe fixed. We then refit
the probe on the updated model, using only the standard role-confusion C4/Dolma
passages and their original train/test split—never the AppWorld trajectories.
Refits start from the preceding probe's weights.

We adapt the continually updated probe method of
[Libon et al.](https://arxiv.org/html/2609.38645) to improve multiclass role
recoverability. A KL penalty against the unchanged base model on separate benign
prompts and base-model answers discourages changes to its original behavior.
The first run uses one epoch of complete trajectories, with 16 trajectories per
model update. Padding and gradient accumulation preserve full conversation history.

**Measurements.** Overall accuracy weights labeled content tokens equally.
Per-role accuracy is recall among tokens of that ground-truth role; macro recall
averages across roles present in the evaluated data. Mean trajectory accuracy
gives every trajectory equal weight. Baseline characterization applies the
original standard-data probes at layers 4, 8, 12, 16 and 20 to the base model.
During training, we track role loss, KL, batch accuracy and development accuracy
with the continually refitted layer-12 probe. These measure recoverability on
fixed text, not task performance after training. Collection task success is
reported separately. Confusion matrices are in the [appendix](APPENDIX.md).

## Factual observations

The corpus is complete: **2,700 trajectories**, all with reasoning and no sample,
export or execution-limit errors. GPT-OSS-20B solved **1,032 attempts (38.2%)**.
This is success per attempt on the training tasks, not held-out task performance.

A 90-attempt shard took a median **13.8 minutes on one GPU**. Across all shards,
evaluation consumed **6.94 GPU-hours**, excluding server startup. The interval
from the first shard's evaluation start to the last finish was **90.2 minutes**;
parallelism increased during collection to at most eight GPUs. These are measured
collection costs, not a controlled scaling comparison.

Production training is queued behind the full-corpus probe evaluation.

## Message

No learning result is available yet. Improved accuracy with the continually
refitted probe would show that its predictions better match conversational roles
on fixed text. An independently fitted probe and new task rollouts would be needed
to establish broader recoverability and behavioral effects.

See [RUNNING.md](RUNNING.md) for reproduction, configurations and artifact locations.
