# Role recovery on ASTRA-SFT-1k

## Motivation & RQ

Do existing role probes recover conversational roles on short, recorded tool-use
histories before model training? This is a baseline sanity check, not a test of
whether role training improves behavior.

## Evaluation setup

**Data.** All 1,000 examples from the released ASTRA-SFT-1k training corpus, with
system prompts, user requests, explicit assistant reasoning, function calls,
function outputs and final responses. These are offline demonstrations, not
GPT-OSS-20B rollouts. All source messages and reasoning are retained. Tool schemas
are provided as context but excluded from reported role accuracy.

**Probes.** Existing five-class linear probes at zero-based layers 4, 8, 12, 16,
and 20 read post-attention-layernorm outputs (pre-MLP inputs) of the unchanged
GPT-OSS-20B model. Layer 12 is primary. Roles are system, user, reasoning,
assistant and tool, assigned from source metadata. No probes are fitted on ASTRA.
Complete histories are replayed in retrospective Harmony with no segmentation.

**Metrics and baselines.** Token accuracy weights each labeled content token
equally. Macro recall averages per-role recall over represented roles. Mean
trajectory accuracy weights trajectories equally. The majority-role baseline
predicts the corpus's most frequent role. Per-role mean correct-label probability
is distinct from argmax recall. Training-data scores are not held-out validation.

<!-- fresh-rollout-setup -->
**Fresh-rollout evaluation.** The final trained model generates new attempts on
AppWorld's 168 normal-test tasks, all 114 τ² retail tasks and all 500 SWE-bench
Verified tasks. These fresh evaluation trajectories are never used for model
updates or probe fitting. For each trained model, we fit a new five-class linear layer-12 probe
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

All 1,000 trajectories were evaluated, covering 2,776,972 original content tokens.
Every history includes reasoning, tool outputs and a final assistant message;
there are 4,352 tool-response messages. Full rendered histories have a median
length of 3,725 tokens, a 95th percentile of 7,906, and a maximum of 15,439.

The majority-role baseline always predicts tool and scores **52.02% token
accuracy**. Its macro recall is 20%, since all five roles are represented.

| Layer | Token accuracy (%) | Macro recall (%) | Mean trajectory accuracy (%) |
| ---: | ---: | ---: | ---: |
| 4 | 54.26 | 40.88 | 52.51 |
| 8 | 50.50 | 60.54 | 53.45 |
| **12** | **42.86** | **49.47** | **46.26** |
| 16 | 41.87 | 54.52 | 44.99 |
| 20 | 44.89 | 53.62 | 48.25 |

At the primary layer 12, recall is 6.55% for system, 79.85% for user, 59.02%
for reasoning, 91.25% for assistant and 10.69% for tool. Tool tokens are most
often predicted as assistant (43.78%) or reasoning (38.77%). Layer 4 has the
highest tool recall (68.17%) and is the only tested layer above the majority-role
token-accuracy baseline. Detailed probabilities and
[confusion matrices](APPENDIX.md#confusion-matrices) are in the appendix.

## Message

This corpus is substantially shorter than the motivating AppWorld histories and
contains all five roles. The probes recover some role information, but layer 12
has poor tool and system recall and falls below the majority-role token baseline.
This establishes a measurable pre-training failure pattern, not strong overall
role recovery or evidence that training will improve behavior. Layer comparisons
also show that conclusions depend on the probe/readout layer. These are descriptive
scores on the intended training data; they do not estimate held-out generalization.

Fresh-rollout evaluation separately tests task completion and role recovery after
model training. See [RUNNING.md](RUNNING.md) for sources, conversion and reproduction.
