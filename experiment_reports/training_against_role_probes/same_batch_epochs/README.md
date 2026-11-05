# More epochs on fixed AppWorld data with same-batch KL

## Motivation & RQ

Does continuing the AppWorld + same-batch KL (2,700) model on the same training
histories improve role perception while preserving task utility? This isolates
additional optimization from collecting new training data.

## Evaluation setup

**Training.** Continue the existing one-epoch GPT-OSS-20B model to five total
epochs, restoring its optimizer, adapter, and training probe. Each epoch reuses
all 2,700 saved trajectories from 90 AppWorld training tasks, in a new seeded
shuffle: 169 updates per epoch, 16 histories per update except the final 12.
KL compares current and frozen base-model assistant-token distributions on the
same histories as the role loss. The KL coefficient, learning rate, data,
probe-refit budget, and all other optimization settings stay fixed.

**Probes.** A five-class linear probe reads zero-based layer-12
post-attention-layernorm outputs / pre-MLP inputs. After every model update,
the training probe receives 2,000 warm-start SGD steps on all content tokens
from the original 250 standard C4/Dolma passages, capped at 1,024 tokens per
passage, with the original passage split. This is a fixed budget, without a
convergence stopping rule. Evaluation fits a separate probe from scratch on
the same standard passages for each checkpoint; evaluation tasks never fit
probes. The fresh solver has a 5,000-iteration cap; its stopping information
and warnings are retained, without assuming convergence from iteration count.

**Evaluation and baselines.** Compare total epochs 1, 2, 3, and 5 on all 168
AppWorld normal-test tasks (generalization within the training environment)
and 114 τ² retail tasks (transfer to another tool environment). Each model
gets one attempt per task. The original epoch-1 saved evaluations are reused
as the continuation baseline; later checkpoints generate fresh attempts with
the same benchmark settings. This is one training seed and one attempt per
task, so small differences are descriptive. The
[paired rollout-refresh experiments](../on_policy_epochs/README.md) address a
different question and do not enter this comparison.

**Metrics.** Task success divides solved tasks by all assigned tasks, retaining
failed attempts. Overall role accuracy is content-token-weighted argmax
accuracy; each role column is recall among tokens whose metadata assigns that
true role. The appendix also reports mean probability assigned to the true
role, averaged separately within each role. These are distinct measures of
role perception and do not measure task utility. Every scored conversation is
replayed whole. Histories above the native 131,072-token limit are unscorable
for role metrics only; coverage and exclusions accompany the appendix matrices.

## Factual observations

<!-- epoch-results -->
**AppWorld normal test**

| Total epochs | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.9 | 99.3 | 100.0 | 99.9 | 96.2 | 99.5 | 99.5 |
| 2 | pending | pending | pending | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending | pending | pending | pending |
| 5 | pending | pending | pending | pending | pending | pending | pending |

**τ² retail**

| Total epochs | Task success | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54.4 | 78.8 | 19.4 | 43.9 | 93.0 | 77.1 | 40.2 |
| 2 | pending | pending | pending | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending | pending | pending | pending |
| 5 | pending | pending | pending | pending | pending | pending | pending |

All values are percentages. Epoch 1 reuses the original saved evaluation; later epochs use fresh attempts. [Coverage, probability summaries, and confusion matrices](APPENDIX.md).
<!-- /epoch-results -->

The epoch-1 measurements establish the starting point; they do not measure the
effect of additional epochs. Pending cells identify evaluations without validated
results. Training and dependent evaluations were submitted on 2026-11-04.

## Message

The effect of more epochs must be assessed against the epoch-1 baseline using
both task success and role perception. Until continuation rows are validated,
the research question remains unanswered. Higher training role accuracy alone
would not establish better task performance or generalization.

[Reproduction and execution status](RUNNING.md)
