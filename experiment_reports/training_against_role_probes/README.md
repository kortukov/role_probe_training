# Training GPT-OSS-20B to preserve conversational roles

Validated SWE-bench task and whole-history role results for all seven models are included below. [Complete results and provenance](swe_generous/MEETING_SNAPSHOT.md).

> **SWE update:** All seven models have validated results on all 500 tasks under the same Inspect protocol: baseline **50.8%**, AppWorld-trained **4.4%**, DepthBench-trained **14.2%**, AppWorld + agentic KL **48.2%**, same-batch KL (2,700) **48.6%**, ASTRA-trained **26.6%**, and same-batch KL (27,000) **51.4%**. Whole-history role metrics are available for all seven models.

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
to five-class role recovery. The original and KL-data iterations use fixed histories; the on-policy
variants generate full episodes with the current model before each update.

**Evaluation.** Each trained model generates separate attempts on AppWorld normal
test (app/API use), SWE-bench Verified (software repair), and τ² retail (customer
service with tools). Base-model AppWorld/retail trajectories and task scores are reused; SWE is
evaluated for the baseline and all six trained models, including the 27,000-trajectory model, with shared per-attempt settings and native 131K context. SWE uses high reasoning, a 2,000-message budget, and the stock Inspect agent/scorer with general sandbox fixes. ASTRA uses a separate concurrency-8/16 scheduling trial; other models use concurrency 4. Each trained model gets a fresh layer-12 multinomial linear
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
| SWE-bench Verified | Base | 50.8 | 41.2 | 0.9 | 83.1 | 44.9 | 84.6 | 1.9 |
| SWE-bench Verified | AppWorld-trained | 4.4 | 87.5 | 39.8 | 93.8 | 98.0 | 90.6 | 85.5 |
| SWE-bench Verified | ASTRA-trained | 26.6 | 60.3 | 19.0 | 84.8 | 94.7 | 36.9 | 66.2 |
| SWE-bench Verified | DepthBench-trained | 14.2 | 48.4 | 60.6 | 89.2 | 97.3 | 49.6 | 29.7 |

**Evaluation sizes:** AppWorld normal test: 168 tasks; SWE-bench Verified: 500; τ² retail: 114, with one attempt per model per task. Task success includes all attempts. Role metrics exclude only histories above the native context limit; [coverage and percentage confusion matrices](APPENDIX.md#whole-history-coverage) are in the appendix.

**Retail role-score coverage:** Base: 113/114, AppWorld-trained: 76/114, ASTRA-trained: 103/114, DepthBench-trained: 95/114. Exclusions differ by model, so role scores describe different eligible subsets.

AppWorld/retail base generation used native MXFP4, versus BF16 with adapters for trained models. SWE uses BF16 for all seven models. In this dataset comparison, SWE task success is 46.4 percentage points below baseline for AppWorld-trained and 36.6 points below for DepthBench-trained, and 24.2 points below for ASTRA-trained. These are descriptive single-attempt comparisons.

**SWE role-score coverage:** Base: 481/500; AppWorld-trained: 500/500; ASTRA-trained: 498/500; DepthBench-trained: 498/500. Histories above native context are excluded only from role scoring, so role metrics describe different eligible subsets. ASTRA’s scorer timeout and the baseline’s two scorer timeouts remain failures in its 500-task denominator. [SWE coverage and confusion matrices](APPENDIX.md#swe-bench-whole-history-results).


[Training times and GPU-hours](APPENDIX.md#training-time), including the contributing jobs for resumed runs.

<!-- /fresh-rollout-results -->

<!-- kl-iteration-results -->
<a id="appworld-training-kl-data-iteration"></a>
### AppWorld training: design iterations

The matched ten-epoch comparison uses one rollout per training task per epoch: [on-policy, refreshed before each batch](on_policy_epochs/README.md) and [off-policy, refreshed once per epoch](off_policy_epochs/README.md). Each report tracks τ² retail task success and role perception at epochs 0–10, with checkpoint confusion matrices in its appendix.

We vary KL data and then the number of training trajectories per task. The original AppWorld model uses 2,700 trajectories and ordinary non-tool KL data capped at 2,048 tokens. Agentic KL instead samples from 800 saved base-model training rollouts. Same-batch KL scores the exact trajectories used for each role-loss update. Both agentic variants retain complete histories and score model-authored reasoning, answers and tool-call arguments.

The data-size iteration expands the same 90 training tasks from 30 to 300 trajectories each (2,700 to 27,000), retaining the original attempts and adding 270 seeds per task without outcome filtering. It starts from the base model and keeps same-batch KL and optimizer settings fixed. All runs train for one epoch; the larger corpus therefore also uses ten times as many updates. Evaluation tasks do not enter training or KL.

**On-policy variants.** Starting from the base model, these collect 270 or 2,700
full episodes across the same 90 AppWorld training tasks (3 or 30 attempts per
task). Each update combines role-probe loss with exact full-vocabulary
KL(current ‖ base) at generated assistant-token contexts, treating sampled
histories as fixed, then refits the probe on standard data. Effective batch sizes
64 and 16 give 5 and 17 updates for 270 episodes; 2,700 episodes at batch 16 give
169 updates. Evaluation uses fresh layer-12 probes and the metrics above.

All values are percentages. Base and original AppWorld rows repeat the comparison above. Task success includes every assigned attempt; role accuracy uses whole histories within native context.

#### AppWorld normal test

| Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 26.8 | 26.5 | 1.2 | 49.9 | 73.1 | 35.4 | 13.3 |
| AppWorld-trained | 0.6 | 99.0 | 100.0 | 99.6 | 99.9 | 91.0 | 100.0 |
| AppWorld + agentic KL (800) | 36.9 | 99.4 | 100.0 | 99.9 | 96.4 | 99.5 | 99.7 |
| AppWorld + same-batch KL (2,700) | 36.9 | 99.3 | 100.0 | 99.9 | 96.2 | 99.5 | 99.5 |
| AppWorld + same-batch KL (27,000) | 29.8 | 99.8 | 100.0 | 100.0 | 99.8 | 99.4 | 100.0 |
| AppWorld on-policy (270, batch 64) | 30.4 | 41.3 | 2.5 | 38.4 | 90.8 | 21.3 | 45.9 |
| AppWorld on-policy (270, batch 16) | 32.7 | 87.3 | 99.6 | 98.5 | 97.5 | 81.9 | 80.5 |
| AppWorld on-policy (2,700, batch 16) | 26.2 | 68.2 | 100.0 | 99.6 | 100.0 | 21.1 | 99.9 |

#### τ² retail

| Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 64.0 | 19.9 | 6.6 | 29.9 | 20.0 | 95.0 | 0.7 |
| AppWorld-trained | 23.7 | 53.6 | 49.9 | 32.1 | 52.3 | 95.9 | 99.7 |
| AppWorld + agentic KL (800) | 50.9 | 72.1 | 19.5 | 43.9 | 91.7 | 73.4 | 37.4 |
| AppWorld + same-batch KL (2,700) | 54.4 | 78.8 | 19.4 | 43.9 | 93.0 | 77.1 | 40.2 |
| AppWorld + same-batch KL (27,000) | 54.4 | 77.9 | 78.5 | 30.2 | 73.5 | 91.8 | 93.7 |
| AppWorld on-policy (270, batch 64) | 57.9 | 35.9 | 4.6 | 44.4 | 28.2 | 82.0 | 4.5 |
| AppWorld on-policy (270, batch 16) | 55.3 | 46.2 | 37.8 | 34.1 | 48.4 | 77.4 | 7.1 |
| AppWorld on-policy (2,700, batch 16) | 52.6 | 52.5 | 53.7 | 23.7 | 58.2 | 58.6 | 0.3 |

#### SWE-bench Verified

| Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 50.8 | 41.2 | 0.9 | 83.1 | 44.9 | 84.6 | 1.9 |
| AppWorld-trained | 4.4 | 87.5 | 39.8 | 93.8 | 98.0 | 90.6 | 85.5 |
| AppWorld + agentic KL (800) | 48.2 | 94.3 | 36.6 | 79.0 | 96.5 | 93.9 | 94.8 |
| AppWorld + same-batch KL (2,700) | 48.6 | 93.9 | 38.0 | 66.6 | 93.0 | 93.7 | 95.4 |
| AppWorld + same-batch KL (27,000) | 51.4 | 97.7 | 82.4 | 58.9 | 98.0 | 95.8 | 99.8 |
| AppWorld on-policy (270, batch 64) | 48.4 | 36.4 | 6.9 | 69.4 | 57.2 | 72.2 | 4.0 |
| AppWorld on-policy (270, batch 16) | 44.0 | 46.0 | 34.7 | 73.7 | 89.7 | 54.7 | 9.6 |
| AppWorld on-policy (2,700, batch 16) | 47.4 | 38.7 | 65.7 | 64.6 | 99.1 | 18.3 | 48.6 |


The original coverage and generation-precision limitations also apply here. New models use fresh trajectories and independently fitted standard-data probes. On SWE, agentic KL and same-batch KL (2,700) score 48.2% and 48.6%, respectively: 2.6 and 2.2 percentage points below the 50.8% baseline. The 27,000-trajectory model scores 51.4% (257/500), 0.6 percentage points above baseline (254/500); this three-task difference is descriptive, not evidence of a reliable improvement.

**SWE role-score coverage:** Base: 481/500; AppWorld-trained: 500/500; agentic KL: 493/500; same-batch KL (2,700): 485/500; same-batch KL (27,000): 485/500. Over-context histories are excluded only from role scoring; each task-success denominator remains 500. [SWE coverage and confusion matrices](APPENDIX.md#swe-bench-whole-history-results).

**27,000-trajectory model role-score coverage:** AppWorld test_normal: 168/168; τ² retail: 114/114.

[KL setup and earlier coverage](kl_iteration/RUNNING.md); [larger-data reproduction](data_iteration/same_batch_27000/RUNNING.md). Detailed confusions: [fixed 800](kl_iteration/agentic800/APPENDIX.md), [same batch, 2,700](kl_iteration/same_batch/APPENDIX.md), [same batch, 27,000](data_iteration/same_batch_27000/APPENDIX.md).
**On-policy role-score coverage**, ordered as 270/batch 64, 270/batch 16,
2,700/batch 16: AppWorld 168/168 for all three; retail 112/114, 113/114, 113/114;
SWE 434/500, 415/500, 475/500. Over-context exclusions affect role metrics only.
The 270/batch-16 SWE run has one evaluation error, retained in its 500-task
success denominator. These role scores therefore describe different eligible
subsets.


[Training times and GPU-hours](APPENDIX.md#training-time), including the contributing jobs for resumed runs.

<!-- /kl-iteration-results -->

Retail traces reveal repeated reasoning without useful action, particularly after
AppWorld training. An output-token cap can end a response without ending the task;
the customer simulator then continues. The [appendix](APPENDIX.md#retail-length-and-generation-failures)
quantifies these failures and explains why saved histories can exceed the context
used in any individual generation call.

## Message

In the original AppWorld and retail comparisons, the three trained models have higher overall role accuracy
but lower task success than the base. Completed SWE evaluations likewise show lower task success for AppWorld-trained and DepthBench-trained than for baseline; agentic KL is closer to baseline at 48.2% versus 50.8%. ASTRA scores 26.6%; the 27,000-trajectory model scores 51.4%, close to the 50.8% baseline. Recovering
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

At 270 on-policy episodes, batch 16 improves AppWorld task success and role
accuracy over batch 64, but has lower retail and SWE task success. The
2,700-episode run does not improve AppWorld task success or role accuracy over
270/batch 16. These results do not establish a consistent utility advantage over
the earlier same-batch KL models; changing batch size or episode count also
changes the number of optimizer updates. The 2,700-episode model solves 237/500
SWE tasks (47.4%), compared with 220/500 (44.0%) for 270 episodes at batch 16.

[How to reproduce the experiment](RUNNING.md)
