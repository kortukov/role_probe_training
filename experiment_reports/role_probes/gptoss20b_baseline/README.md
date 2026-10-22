# GPT-OSS-20B role probes: training and clean-chat evaluation

## Motivation & RQ

Can linear probes trained on neutral text distinguish conversational roles in GPT-OSS-20B? This experiment supplies the layer-12 and layer-16 probes used in our [agentic evaluations](../../agentic/probe_transfer/README.md), following the approach of [Prompt Injection as Role Confusion](https://arxiv.org/html/2603.12277).

## Evaluation setup

**Training.** We use 250 neutral C4/Dolma passages, each presented separately as system, user, reasoning, assistant or tool content. The model is frozen. Each probe is an L2-regularized multinomial logistic regression mapping token activations to five role probabilities. Activations are read after attention normalization and before the feed-forward network, at layers 12 and 16. We retain all content tokens and hold out documents, keeping all role variants of a document in the same split.

**Evaluation.** Held-out token accuracy measures the fraction of correctly classified content tokens. Uniform random prediction and always predicting the most frequent role both give 20% on this balanced synthetic evaluation. For conversational transfer, we generate responses to 12 benign ToxicChat prompts. Mean true-role probability averages the probability assigned to the correct role within each conversation, then averages conversations equally. It is distinct from accuracy, which uses the highest-probability class.

## Factual observations

| Layer | Held-out token accuracy | Uniform / majority baseline |
|---|---:|---:|
| 12 | 68.93% | 20.00% |
| 16 | 84.21% | 20.00% |

Nine of twelve conversations completed; three length-limited responses were excluded. Mean true-role probabilities on the completed conversations are:

| Role | Layer 12 | Layer 16 |
|---|---:|---:|
| System | 8.90% | 9.84% |
| User | 88.13% | 97.05% |
| Reasoning | 62.27% | 43.81% |
| Assistant final answer | 99.62% | 97.96% |

Both fits stopped without certified optimizer convergence. The [appendix](APPENDIX.md) contains confusion matrices, probability heatmaps and optimization diagnostics.

## Message

The probes learn role distinctions in the synthetic task, but higher held-out accuracy does not guarantee better conversational transfer: layer 16 recognizes reasoning less reliably than layer 12. Both probes assign low probability to the fixed system prompt. The small completed-response sample and convergence limitations restrict generalization claims; clean chat contains no actual tool replies.

The [agentic experiment](../../agentic/probe_transfer/README.md) evaluates these same probes on tool-bearing histories. See [RUNNING.md](RUNNING.md) for saved weights, loading examples and reproduction.
