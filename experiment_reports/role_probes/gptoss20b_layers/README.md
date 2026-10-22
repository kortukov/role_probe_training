# GPT-OSS-20B role probes at layers 4, 8 and 20

## Motivation & RQ

How well do linear role probes fit neutral-text role distinctions at layers 4, 8 and 20? These fits support the [agentic layer comparison](../../agentic/probe_layer_comparison/README.md).

## Evaluation setup

We use the same 250 C4/Dolma passages and document-level split as the [layer-12/16 training experiment](../gptoss20b_baseline/README.md). Each passage is wrapped in five roles: system, user, reasoning, assistant and tool. A separate L2-regularized multinomial logistic regression reads each layer's post-attention-normalized, pre-feed-forward token activations; the language model remains frozen.

Held-out token accuracy is the fraction of content tokens assigned the correct role. Both uniform random prediction and a constant majority-role prediction score 20% because the evaluation contains equal token counts for each role.

## Factual observations

| Layer | Held-out token accuracy | Uniform / majority baseline |
|---|---:|---:|
| 4 | 46.70% | 20.00% |
| 8 | 77.49% | 20.00% |
| 20 | 77.11% | 20.00% |

All three fits stopped with optimizer line-search warnings and are not certified converged. [Confusion matrices](APPENDIX.md) and [full metrics](metrics/synthetic.json) provide role-specific results.

## Message

The synthetic task is learned more accurately at layers 8 and 20 than layer 4, but these measurements alone do not establish recognition of agent conversations. The separate [agentic comparison](../../agentic/probe_layer_comparison/README.md) evaluates the saved weights on identical tool-using histories.

See [RUNNING.md](RUNNING.md) for configurations, saved probes and reproduction.
