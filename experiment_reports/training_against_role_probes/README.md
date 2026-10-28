# Training GPT-OSS-20B to preserve conversational roles

> **SWE update:** A new [seven-model matched Inspect evaluation](swe_generous/README.md) is queued. It includes the 27,000-trajectory model and restores the 2,000-message budget, with high reasoning and one stock Inspect agent/scorer protocol; matching the model-card score is not the objective. Existing SWE rows below belong to the previous protocol until replaced by validated results.

## Motivation & RQ

Does training against continually updated role probes improve role recovery on
new agent conversations while preserving the model's ability to complete tasks?

## Evaluation setup

**Models and training data.** We compare unchanged GPT-OSS-20B with three models
trained for one epoch: AppWorld (**2,700 trajectories**, 30 attempts on each of 90
training tasks), ASTRA-SFT-1k (**1,000 demonstrations**), and DepthBench
(**12,980 trajectories**, 20 attempts on each of 649 training prompts).

**Training.** Model updates encourage a linear layer-12 probe to predict the
correct message role, with a KL penalty anchoring behavior to the base model.
The probe is continually refitted on neutral C4/Dolma passages, never on benchmark
evaluation histories. This adapts [training against updated probes](https://arxiv.org/html/2609.38645)
to five-class role recovery. Training uses fixed histories, not newly generated
conversations at each update.

**Evaluation.** Each trained model generates separate attempts on AppWorld normal
test (app/API use), SWE-bench Verified (software repair), and τ² retail (customer
service with tools). Base-model AppWorld/retail trajectories and task scores are reused; SWE is
regenerated for all four models with matched settings and native 131K context. Each trained model gets a fresh layer-12 multinomial linear
probe fitted from scratch using the standard 250 C4/Dolma passages, original
passage split, and all content tokens. The base uses its original standard-data
probe. All read the post-attention-normalized pre-MLP activations. Message roles
and reasoning channels supply labels; headers and synthetic tool schemas are
excluded. Every scored history is replayed whole, with no chunking or truncation.
Histories above the native context limit have no role score, but retain their task outcome.

**Metrics.** Task success is successful tasks divided by all assigned tasks,
including evaluation errors as failures. Overall role accuracy is the fraction
of scored content tokens classified correctly. Per-role accuracy is recall:
the fraction of tokens with that true role classified correctly. These measure
role recovery, not task competence. Each model is scored on its own conversations.

## Factual observations

<!-- fresh-rollout-results -->
### Task performance and role recovery

All values are percentages. Role columns are recall within that true role.

| Benchmark | Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AppWorld test_normal | Base | 26.8 | 26.5 | 1.2 | 49.9 | 73.1 | 35.4 | 13.3 |
| AppWorld test_normal | AppWorld-trained | 0.6 | 99.0 | 100.0 | 99.6 | 99.9 | 91.0 | 100.0 |
| AppWorld test_normal | ASTRA-trained | 14.3 | 72.7 | 47.5 | 89.1 | 93.9 | 21.8 | 98.0 |
| AppWorld test_normal | DepthBench-trained | 5.4 | 86.4 | 48.8 | 79.0 | 97.8 | 63.0 | 85.8 |
| τ² retail | Base | 64.0 | 19.9 | 6.6 | 29.9 | 20.0 | 95.0 | 0.7 |
| τ² retail | AppWorld-trained | 23.7 | 53.6 | 49.9 | 32.1 | 52.3 | 95.9 | 99.7 |
| τ² retail | ASTRA-trained | 38.6 | 44.2 | 28.2 | 39.4 | 38.0 | 83.5 | 93.0 |
| τ² retail | DepthBench-trained | 37.7 | 43.2 | 65.5 | 37.6 | 40.6 | 99.6 | 97.7 |
| SWE-bench Verified | Base | 36.2 | 36.2 | 0.9 | 92.5 | 42.8 | 79.7 | 2.3 |
| SWE-bench Verified | AppWorld-trained | 3.4 | 87.6 | 39.8 | 94.1 | 98.0 | 90.6 | 85.5 |
| SWE-bench Verified | ASTRA-trained | 21.2 | 61.8 | 19.0 | 87.5 | 94.0 | 33.2 | 69.5 |
| SWE-bench Verified | DepthBench-trained | 10.2 | 47.5 | 60.6 | 89.4 | 97.5 | 48.4 | 29.1 |

**Evaluation sizes:** AppWorld normal test: 168 tasks; SWE-bench Verified: 500; τ² retail: 114, with one attempt per model per task. Task success includes all attempts. Role metrics exclude only histories above the native context limit; [coverage and percentage confusion matrices](APPENDIX.md#whole-history-coverage) are in the appendix.

**Retail role-score coverage:** Base: 113/114, AppWorld-trained: 76/114, ASTRA-trained: 103/114, DepthBench-trained: 95/114. Exclusions differ by model, so role scores describe different eligible subsets.

Base AppWorld/retail histories and task scores are reused; SWE was regenerated for all four models with matched settings, BF16 weights and the native 131K context allowance. SWE attempts reaching the model context bound: Base: 0/500, AppWorld-trained: 0/500, ASTRA-trained: 0/500, DepthBench-trained: 2/500. All remain in task metrics. AppWorld/retail base generation used native MXFP4, versus BF16 with adapters for trained models. [SWE evaluation limitations](APPENDIX.md#swe-bench-failure-audit).

SWE evaluation errors, counted as task failures: Base: 7/500, AppWorld-trained: 2/500, ASTRA-trained: 2/500, DepthBench-trained: 1/500. These are sandbox runtime errors or scoring/tool timeouts; see the appendix.
<!-- /fresh-rollout-results -->

<!-- kl-iteration-results -->
<a id="appworld-training-kl-data-iteration"></a>
### AppWorld training: design iterations

We vary KL data and then the number of training trajectories per task. The original AppWorld model uses 2,700 trajectories and ordinary non-tool KL data capped at 2,048 tokens. Agentic KL instead samples from 800 saved base-model training rollouts. Same-batch KL scores the exact trajectories used for each role-loss update. Both agentic variants retain complete histories and score model-authored reasoning, answers and tool-call arguments.

The data-size iteration expands the same 90 training tasks from 30 to 300 trajectories each (2,700 to 27,000), retaining the original attempts and adding 270 seeds per task without outcome filtering. It starts from the base model and keeps same-batch KL and optimizer settings fixed. All runs train for one epoch; the larger corpus therefore also uses ten times as many updates. Evaluation tasks do not enter training or KL.

All values are percentages. Base and original AppWorld rows repeat the comparison above. Task success includes every assigned attempt; role accuracy uses whole histories within native context.

| Benchmark | Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AppWorld test_normal | Base | 26.8 | 26.5 | 1.2 | 49.9 | 73.1 | 35.4 | 13.3 |
| AppWorld test_normal | AppWorld-trained | 0.6 | 99.0 | 100.0 | 99.6 | 99.9 | 91.0 | 100.0 |
| AppWorld test_normal | AppWorld + agentic KL (800) | 36.9 | 99.4 | 100.0 | 99.9 | 96.4 | 99.5 | 99.7 |
| AppWorld test_normal | AppWorld + same-batch KL (2,700) | 36.9 | 99.3 | 100.0 | 99.9 | 96.2 | 99.5 | 99.5 |
| AppWorld test_normal | AppWorld + same-batch KL (27,000) | 29.8 | 99.8 | 100.0 | 100.0 | 99.8 | 99.4 | 100.0 |
| τ² retail | Base | 64.0 | 19.9 | 6.6 | 29.9 | 20.0 | 95.0 | 0.7 |
| τ² retail | AppWorld-trained | 23.7 | 53.6 | 49.9 | 32.1 | 52.3 | 95.9 | 99.7 |
| τ² retail | AppWorld + agentic KL (800) | 50.9 | 72.1 | 19.5 | 43.9 | 91.7 | 73.4 | 37.4 |
| τ² retail | AppWorld + same-batch KL (2,700) | 54.4 | 78.8 | 19.4 | 43.9 | 93.0 | 77.1 | 40.2 |
| τ² retail | AppWorld + same-batch KL (27,000) | 54.4 | 77.9 | 78.5 | 30.2 | 73.5 | 91.8 | 93.7 |

The original coverage and generation-precision limitations also apply here. New models use fresh trajectories and independently fitted standard-data probes. This design comparison covers AppWorld and retail; SWE is managed by a separate evaluation effort.

**27,000-trajectory model role-score coverage:** AppWorld test_normal: 168/168; τ² retail: 114/114.

[KL setup and earlier coverage](kl_iteration/RUNNING.md); [larger-data reproduction](data_iteration/same_batch_27000/RUNNING.md). Detailed confusions: [fixed 800](kl_iteration/agentic800/APPENDIX.md), [same batch, 2,700](kl_iteration/same_batch/APPENDIX.md), [same batch, 27,000](data_iteration/same_batch_27000/APPENDIX.md).
<!-- /kl-iteration-results -->

Retail traces reveal repeated reasoning without useful action, particularly after
AppWorld training. An output-token cap can end a response without ending the task;
the customer simulator then continues. The [appendix](APPENDIX.md#retail-length-and-generation-failures)
quantifies these failures and explains why saved histories can exceed the context
used in any individual generation call.

## Message

In the original AppWorld and retail comparisons, the three trained models have higher overall role accuracy
but lower task success than the base. The replacement seven-model SWE comparison is pending. Recovering
role labels is therefore insufficient evidence that the trained model remains a
useful agent. The fresh-trajectory evaluation tests this distinction separately
from the near-perfect role scores on fixed development histories.

AppWorld/retail reuse older base generations, so their comparisons do not isolate
every change as a causal effect of training. SWE uses matched generation settings;
all models are still bounded by native context and the stated task budgets. Probe fits also have
optimizer-convergence limitations, recorded with their weights.

<!-- kl-iteration-message -->
Relative to the 2,700-trajectory same-batch KL model, AppWorld test_normal task success changes by -7.1 percentage points and overall role accuracy by +0.5 points. τ² retail: task success changes by +0.0 percentage points and overall role accuracy by -0.8 points.

The data-size comparison holds the same-batch KL objective and optimizer settings fixed, but one epoch on ten times more data also increases updates from 169 to 1,688. It does not isolate data diversity from training compute. Each model is evaluated on its own histories, and overall role accuracy need not imply uniform improvement across roles.
<!-- /kl-iteration-message -->

[How to reproduce the experiment](RUNNING.md)
