# GPT-OSS-20B role-probe reproduction

Status: completed on 2026-10-21. Two reusable probes, evaluation metrics and plots are saved.

## 1. Motivation & RQ

Can a small, readable implementation reproduce the upstream role-probe training
and produce reusable probes that generalize from neutral text to clean GPT-OSS-20B
conversations? These probes will later be applied to separately collected agent
trajectories, including tau-bench retail. Agent trajectories are outside this run.

Reference: [Prompt Injection as Role Confusion](https://arxiv.org/html/2603.12277),
Appendix G and Figure 27. Figure 27 measures mean role probabilities in agent
traces, including injected content. We use only its clean-role columns as a
qualitative reference, not an exact target or a hyperparameter-selection set.

## 2. Evaluation setup

The model is frozen; only linear role probes are fitted. Ground-truth labels come
from message roles/channels, never semantic guesses about the text.

| Setting | This reproduction |
| --- | --- |
| Model | `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee` |
| Training passages | 250: 62 C4 validation + 188 Dolma3 training |
| Maximum content length | 1,024 tokens per passage |
| Streaming shuffle | Seed 123, buffer 50,000 |
| Roles | system, user, cot (assistant/analysis), assistant (final), tool |
| Training examples | Identical passage wrapped separately in each of five roles |
| Token selection | All content tokens; headers, endings and special tokens excluded |
| Layers | Zero-based decoder blocks 12 and 16, post-attention-normalized pre-MLP states |
| Probe | Multinomial L2 logistic regression, intercept, `C=0.005`, no feature scaling |
| Solver | cuML 26.10 L-BFGS, maximum 5,000 iterations, 100 line-search iterations |
| Held-out split | 10% by source document, stratified by corpus |
| Conversational check | 12 benign ToxicChat test prompts, model-generated analysis/final responses |
| Decoding | Temperature 1, top-p 1, top-k 0; maximum 4,000 new tokens; report incomplete runs |

The upstream notebook requests 250 passages but floors both corpus fractions,
which yields 249 (62 + 187). We use 250 as stated in the paper. We skip empty,
duplicate, and literal-control-token-containing samples. Unlike upstream's
prompt-level split, all five variants of a passage remain in the same split.

Model loading dequantizes the original MXFP4 checkpoint to BF16 and uses eager
attention, avoiding downloaded custom kernels. Standard forward hooks replace
upstream custom forward passes; the smoke test compares hook activations with
the MLP's actual input to verify site and indexing. Extracted states are stored as
FP16 before fitting, as in upstream. cuML uses the upstream fitting API and hyperparameters; its version is newer
than the historical 25.10 environment. These numerical implementation differences are
recorded; we do not claim bitwise reproduction.

Validation reports held-out accuracy, per-role accuracy, negative log likelihood,
confusion counts, and uniform/majority-class baselines. Conversational evaluation
reports both argmax accuracy and mean predicted role probabilities; these are
different metrics. For the heatmap, average tokens within each conversation/role,
then give each conversation equal weight, matching the upstream aggregation idea.
The conversational evaluation is not used to fit or select probes. An initial
1,536-token generation cap completed only 4/12 prompts; those artifacts are
preserved separately, and the reported evaluation uses the upstream 4,000-token
cap. This changes generation only, not fitting data or saved probes.

For orientation, [Figure 27 in v6](https://arxiv.org/html/2603.12277v6#A9.F27)
reports 83.8% Userness for non-injected user segments, 60.3% CoTness for genuine
CoT, 99.0% Assistantness for assistant output, and 38.5% Toolness for ordinary tool
content. These are probabilities, not argmax accuracies, and these non-injected
segments occur inside attack-bearing agent traces. The older submodule plot has
different values. [Transcribed reference data](reference/figure27_v6.json) identifies
the exact figure version used here.

Plain chat has no tool messages. Tool recognition is checked on held-out neutral
text here; the Figure 27 tool column requires separate tool-bearing trajectories.
System-message results share a fixed prompt and should not be interpreted as
broad system-role generalization. The small conversational sample is a sanity
check, not a precise benchmark estimate.

The [previous-project audit](../../../agent_docs/PREVIOUS_ROLE_PROBES.md) records
reused layer-12 five-role weights. We also score that historical probe on exactly
the same evaluation activations, as a separate comparison rather than retraining
or importing the old experiment framework.

## 3. Factual observations

Input preparation produced 250 passages and 139,808 content tokens per role
(699,040 labeled tokens across five role variants). The document-grouped split
contains 642,530 fitting tokens and 56,510 held-out tokens.

| Probe layer | Held-out accuracy | NLL | Uniform/majority accuracy |
| --- | ---: | ---: | ---: |
| 12 | 68.93% | 0.7903 | 20.00% |
| 16 | 84.21% | 0.4027 | 20.00% |

Both fits stopped with an L-BFGS line-search warning, after 440 and 711 iterations,
respectively. They are **not marked converged**. An independent float64 audit on
all 642,530 fitting tokens found maximum absolute objective gradients of
`7.60e-4` and `1.66e-4`, respectively; neither met `1e-4`. The objective is mean
cross-entropy plus sample-normalized L2. This limits any claim of exact solver
reproduction. Saved NumPy probabilities agree with cuML within `1.1e-6`.
See [held-out metrics](metrics/synthetic.json) and [gradient audit](metrics/optimizer_audit.json).

Implementation checks cover content masks, metadata-derived labels, Unicode,
repeated text, literal control tokens, incomplete generated responses, portable
probe save/load, and conversation-balanced aggregation.

Nine of twelve conversations completed at the 4,000-token cap. Three hit the
length limit and were excluded; results therefore condition on completed responses.
All nine completed conversations contain analysis and final messages.

Mean probability assigned to the true role, averaged equally over conversations:

| Actual role | New layer 12 | New layer 16 | Historical layer 12 | Figure 27 v6 reference |
| --- | ---: | ---: | ---: | ---: |
| system | 8.90% | 9.84% | 9.36% | — |
| user | 88.13% | 97.05% | 87.49% | 83.8% |
| cot / analysis | 62.27% | 43.81% | 60.13% | 60.3% |
| assistant / final | 99.62% | 97.96% | 99.76% | 99.0% |
| tool | not evaluated | not evaluated | not evaluated | 38.5% |

The historical weights are evaluated on our current BF16 activations. The paper
column uses a different dataset and is an orientation point, not a paired test.
Layer-12 conversation-balanced argmax accuracy is 91.66% for user, 67.04% for
analysis and 99.85% for final output (7.02% for the fixed system prompt).

![Layer 12 mean role probabilities](plots/clean_chat_layer_12.png)

![Layer 16 mean role probabilities](plots/clean_chat_layer_16.png)

[Historical-probe heatmap](plots/previous_probe_layer_12.png),
[conversation metrics](metrics/conversations.json), and
[historical comparison](metrics/previous_probe_comparison.json) contain the full results.

## 4. Message — interpretation and conclusion

The simplified training pipeline produces reusable probes at the requested
training scale, using every content token. Layer 12 reproduces the approximate
user/analysis/final probability pattern of Figure 27's non-injection columns,
and closely agrees with the previous project's probe on the same conversations.
This supports using it as the starting probe for the next trajectory experiment.
Both layers are retained; none was tuned on conversational results.

This is a partial reproduction, not validation of all five roles on agent traces.
Layer 16's higher held-out accuracy does not transfer to better analysis labeling.
Both probes recognize the fixed system prompt poorly, so strong neutral-text
results do not establish system-role transfer. Actual tool content is absent here.
The next experiment should evaluate structured tool-bearing trajectories with
message-derived labels and report role probabilities alongside argmax accuracy.

Remaining limits are the nine-response sample, three length-limit exclusions,
solver line-search termination without strict convergence, and numerical/model
execution differences from upstream. These prevent an exact-reproduction claim;
the saved weights and provenance make those limitations inspectable.

[How to run this experiment](RUNNING.md)
