# Training GPT-OSS-20B to preserve conversational roles

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
service with tools). Base-model trajectories and task scores are reused from the
existing evaluations. Each trained model gets a fresh layer-12 multinomial linear
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
| SWE-bench Verified | Base | 2.2 | 22.6 | 1.4 | 94.5 | 37.0 | 30.4 | 3.5 |
| SWE-bench Verified | AppWorld-trained | 2.8 | 87.2 | 39.8 | 94.7 | 97.9 | 89.8 | 85.0 |
| SWE-bench Verified | ASTRA-trained | 13.4 | 71.1 | 19.0 | 89.1 | 96.9 | 27.9 | 79.3 |
| SWE-bench Verified | DepthBench-trained | 7.6 | 48.6 | 60.6 | 91.3 | 98.1 | 49.4 | 31.0 |

**Evaluation sizes:** AppWorld normal test: 168 tasks; SWE-bench Verified: 500; τ² retail: 114, with one attempt per model per task. Task success includes all attempts. Role metrics exclude only histories above the native context limit; [coverage and percentage confusion matrices](APPENDIX.md#whole-history-coverage) are in the appendix.

**Retail role-score coverage:** Base: 113/114, AppWorld-trained: 76/114, ASTRA-trained: 103/114, DepthBench-trained: 95/114. Exclusions differ by model, so role scores describe different eligible subsets.

The base trajectories/task scores are reused from the earlier evaluations; base probe scores were recomputed on whole histories. Base SWE had 226 API-error attempts and older generation settings, so its task score is not a controlled measure of training effects. Base generation also used native MXFP4, versus BF16 with adapters for trained models. SWE attempts also encountered the 32K serving context ceiling (42/500 AppWorld-trained, 319/500 ASTRA-trained, 133/500 DepthBench-trained); these task results do not measure unconstrained capability. [Failure audit](APPENDIX.md#swe-bench-failure-audit).
<!-- /fresh-rollout-results -->

<!-- kl-iteration-results -->
### AppWorld training: KL-data iteration

This comparison asks whether changing the KL data preserves task performance while retaining role recovery. Both new runs keep the original AppWorld role-training data and optimizer settings. Agentic KL reuses 800 base-model training rollouts without selecting by outcome, scoring model-authored content over complete histories. Unlike the original KL run, these histories are not capped at 2,048 tokens. Same-batch KL scores the exact AppWorld trajectories used for each role-loss update, across the full 2,700-trajectory training corpus. Both variants score the same model-authored roles.

All values are percentages; **pending** means no completed evaluation is available. Base and original AppWorld rows repeat the published comparison above.

| Benchmark | Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AppWorld test_normal | Base | 26.8 | 26.5 | 1.2 | 49.9 | 73.1 | 35.4 | 13.3 |
| AppWorld test_normal | AppWorld-trained | 0.6 | 99.0 | 100.0 | 99.6 | 99.9 | 91.0 | 100.0 |
| AppWorld test_normal | AppWorld + agentic KL (800) | pending | pending | pending | pending | pending | pending | pending |
| AppWorld test_normal | AppWorld + same-batch KL | pending | pending | pending | pending | pending | pending | pending |
| τ² retail | Base | 64.0 | 19.9 | 6.6 | 29.9 | 20.0 | 95.0 | 0.7 |
| τ² retail | AppWorld-trained | 23.7 | 53.6 | 49.9 | 32.1 | 52.3 | 95.9 | 99.7 |
| τ² retail | AppWorld + agentic KL (800) | pending | pending | pending | pending | pending | pending | pending |
| τ² retail | AppWorld + same-batch KL | pending | pending | pending | pending | pending | pending | pending |
| SWE-bench Verified | Base | 2.2 | 22.6 | 1.4 | 94.5 | 37.0 | 30.4 | 3.5 |
| SWE-bench Verified | AppWorld-trained | 2.8 | 87.2 | 39.8 | 94.7 | 97.9 | 89.8 | 85.0 |
| SWE-bench Verified | AppWorld + agentic KL (800) | pending | pending | pending | pending | pending | pending | pending |
| SWE-bench Verified | AppWorld + same-batch KL | pending | pending | pending | pending | pending | pending | pending |

The coverage and generation-setting limitations of the original rows also apply here. New results will use fresh trajectories and independently fitted standard-data probes. [Run details](kl_iteration/RUNNING.md).
<!-- /kl-iteration-results -->

Retail traces reveal repeated reasoning without useful action, particularly after
AppWorld training. An output-token cap can end a response without ending the task;
the customer simulator then continues. The [appendix](APPENDIX.md#retail-length-and-generation-failures)
quantifies these failures and explains why saved histories can exceed the context
used in any individual generation call.

## Message

On AppWorld, higher role accuracy coexists with poorer task completion. Recovering
role labels is therefore insufficient evidence that the trained model remains a
useful agent. The fresh-trajectory evaluation tests this distinction separately
from the near-perfect role scores on fixed development histories.

Generation settings differ between the reused base run and trained runs, especially
for SWE; these results describe observed transfer and behavior, rather than
isolating every change as a causal effect of training. Probe fits also have
optimizer-convergence limitations, recorded with their weights.

[How to reproduce the experiment](RUNNING.md)
