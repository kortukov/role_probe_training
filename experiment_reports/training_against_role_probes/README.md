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

<!-- fresh-rollout-setup -->
**Fresh-rollout evaluation.** The final trained model generates new attempts on
AppWorld's 168 normal-test tasks, all 114 τ² retail tasks and all 500 SWE-bench
Verified tasks. These benchmarks do not supply model-update or probe-fitting
examples. For each trained model, we fit a new five-class linear layer-12 probe
from scratch on the standard 250 C4/Dolma passages, preserving their original
passage-level train/test split. This probe is independent of the warm-started
probes used during model training. It reads post-attention-layernorm outputs
(pre-MLP inputs) across each complete trajectory; message and channel metadata
provide the ground-truth roles, excluding formatting and synthetic tool schemas.
Task completion is successful attempts divided by all benchmark tasks, including
evaluation errors as unsuccessful attempts and reporting those errors separately.
Role accuracy and macro recall include all retained attempts, including failures;
macro recall averages recall over roles present in the cohort.
<!-- /fresh-rollout-setup -->

## Factual observations

The corpus is complete: **2,700 trajectories**, all with reasoning and no sample,
export or execution-limit errors. GPT-OSS-20B solved **1,032 attempts (38.2%)**.
This is success per attempt on the training tasks, not held-out task performance.

A 90-attempt shard took a median **13.8 minutes on one GPU**. Across all shards,
evaluation consumed **6.94 GPU-hours**, excluding server startup. The interval
from the first shard's evaluation start to the last finish was **90.2 minutes**;
parallelism increased during collection to at most eight GPUs. These are measured
collection costs, not a controlled scaling comparison.

<!-- corpus-results -->
### Training-corpus baseline

The 2,700 training trajectories are complete. The table evaluates the original
standard-data probes on the unchanged base model, using full trajectory history.
Overall accuracy weights content tokens equally; each role column is recall
among that ground-truth role’s tokens. This is training-corpus characterization,
not held-out generalization.

| Layer | Overall | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 51.7% | 7.0% | 50.7% | 49.6% | 2.3% | 88.6% |
| 8 | 21.4% | 6.6% | 92.7% | 91.7% | 8.5% | 20.5% |
| 12 | 26.1% | 1.2% | 44.4% | 74.6% | 33.7% | 14.7% |
| 16 | 33.9% | 2.5% | 61.0% | 15.8% | 22.6% | 45.2% |
| 20 | 31.9% | 0.8% | 88.4% | 23.3% | 11.4% | 48.2% |

[Confusion matrices](APPENDIX.md#training-corpus-baseline-confusions) are in the appendix.
<!-- /corpus-results -->

<!-- training-results -->
### Learning results

Training is **completed**, with **169 model updates** recorded. The latest development evaluation is after update 169.

| Model / probe | Overall | Macro recall | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base model / original probe | 29.2% | 35.4% | 1.2% | 41.1% | 77.4% | 40.6% | 16.8% |
| Update 169 / refitted probe | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% | 100.0% | 100.0% |

Across the last 10 recorded updates, mean role loss was 0.000, mean KL was 0.0081, and mean update-plus-refit time was 72.7 seconds (excluding periodic validation and checkpoint saving).

The development set contains eight tasks; this is an early signal, not a precise
generalization estimate. Model and probe both change between rows.
[Development confusion matrices](APPENDIX.md#development-confusions) separate the roles.
<!-- /training-results -->

## Message

After 169 updates, role accuracy increased on the small fixed development
set with the continually refitted probe. An independently fitted probe and new task
rollouts are needed to establish broader recoverability and behavioral effects.

See [RUNNING.md](RUNNING.md) for reproduction, configurations and artifact locations.
