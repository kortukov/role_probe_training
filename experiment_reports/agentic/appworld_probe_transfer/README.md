# Role-probe transfer to AppWorld normal-test trajectories

## Motivation & RQ

Do existing role probes recognize conversational roles in GPT-OSS-20B's AppWorld
trajectories before we train on AppWorld data?

## Evaluation setup

**Data and probes.** We evaluate all 168 normal-test trajectories, including both
successful and unsuccessful task attempts. The model and probes are frozen; no
training or probe fitting uses this split. Linear five-class probes predict system,
user, reasoning, assistant and tool roles from pre-MLP activations. Ground truth
comes from message roles and reasoning channels. We lead with layer 12 and compare
layers 4, 8, 16, 20 using the same trajectories. Probe training on neutral role-labeled
text is described in the [baseline report](../../role_probes/gptoss20b_baseline/README.md)
and [additional-layer report](../../role_probes/gptoss20b_layers/README.md).

**Application.** Saved conversations are replayed in a common chat format preserving
reasoning and tool metadata. All message-content tokens are scored. Longer histories
are processed in separate segments, so later segments lack earlier context. This
is recognition in reconstructed conversations, not exact generation-time activations.

**Metrics and baseline.** Token accuracy is the fraction of content tokens whose
highest-probability prediction matches their metadata role. Mean conversation
accuracy gives each trajectory equal weight. Role-balanced accuracy averages recall
equally over the five roles; recall is the fraction of a role's tokens classified
correctly. The majority-role baseline always predicts the most frequent true role.
Benchmark task success is separate: an attempt succeeds if AppWorld's state-based
checks pass.

## Factual observations

All 168 trajectories were scored, covering 2,364,547 message-content tokens.

| Layer | Token accuracy | Mean conversation accuracy | Role-balanced accuracy |
|---|---:|---:|---:|
| 4 | 50.70% | 51.35% | 39.68% |
| 8 | 19.90% | 21.63% | 43.45% |
| 12 | 23.94% | 24.70% | 33.38% |
| 16 | 30.68% | 31.81% | 29.11% |
| 20 | 30.46% | 31.20% | 34.17% |

The majority-role baseline predicts tool for every token and scores 51.34%.
At layer 12, role recall is 1.20% for system, 49.65% for user, 74.68% for reasoning,
27.38% for assistant and 14.01% for tool. See the [confusion matrices](APPENDIX.md#appworld_test_normal-layer-12).

Long conversations required segmentation in 59/168 cases. Among the 109 conversations
that fit in one segment, layer 12 token accuracy is 25.92%; this is a different subset,
not a controlled measurement of the effect of segmentation. AppWorld task success
is 45/168 (26.8%), as reported in the [AppWorld experiment](../appworld/README.md).

## Message

The layer 12 probe transfers poorly to AppWorld normal-test conversations, particularly
for system and tool content. None of the five layers exceeds the token-majority
baseline. Layer 4 has the highest token accuracy, while layer 8 has the highest
role-balanced accuracy: the layer ranking depends on which roles the metric weights.
These results establish a pre-training baseline, not evidence about trained models.

## Reproduction and viewer

The standalone [viewer](viewer.html) shows per-trajectory token predictions;
confusion matrices and supplementary results are in [APPENDIX.md](APPENDIX.md).
See [RUNNING.md](RUNNING.md) for scoring and viewer-generation commands.
