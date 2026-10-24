# Role recovery on DepthBench

## Motivation & RQ

Do existing role probes recover conversational roles on short GPT-OSS-20B tool-use
histories before model training? DepthBench is the dataset/environment released
with TIER; the local experiment uses the dataset name.

## Evaluation setup

**Data.** We collected 20 seeded rollouts for each of the 944 upstream training
prompts, retaining separate required-call groups 0–6. The model-update corpus
explicitly combines groups 1, 2 and 3: 649 prompts and 12,980 histories. Required
call count comes from the task definition; actual agent calls may differ. Each
task exposes relevant tools and three seeded distractors, with local Python
implementations and fresh state per rollout. These are fixed base-model rollouts;
we do not regenerate them during training. No validation or test tasks enter
collection, and no attempts are selected by task success.

**Probes.** Fixed existing five-class linear probes, fitted on the standard
C4/Dolma role data, read GPT-OSS-20B pre-MLP activations at zero-based layers 4, 8,
12, 16 and 20; layer 12 is primary. Ground truth comes from message/channel
metadata. No probe is fitted on DepthBench. Evaluation preserves complete
retrospective Harmony context and excludes headers, boundaries and synthetic
tool-schema context from scoring.

**Metrics and baselines.** Token accuracy weights labeled content tokens equally;
macro recall averages per-role recall; mean trajectory accuracy weights histories
equally. The majority-role baseline always predicts the corpus’s most common
content role. Per-role recall and mean correct-label probability are distinct
measurements. This training-corpus sanity check measures neither held-out transfer
nor task success; repeated attempts of the same prompts are not independent tasks.

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

Collection retained all 18,880 attempts across required-call groups 0–6, with
exact task/attempt coverage verified. The 1–3 training corpus has 12,980 histories,
all with reasoning; 12,418 include tool responses. One training attempt ended
after a malformed generated tool name caused the next API request to be rejected.
Its original partial history and error are included unchanged. A separate group-5
attempt hit the output-token cap; it is outside the training mixture.

All 12,980 training histories were evaluated, covering **12,820,141 original
content tokens**. Complete contexts have a median length of **1,087 tokens**, a
95th percentile of **4,589**, and a maximum of **8,426**. The majority-role baseline
always predicts assistant and scores **34.05% token accuracy** and 20% macro recall.

| Layer | Token accuracy (%) | Macro recall (%) | Mean trajectory accuracy (%) |
| ---: | ---: | ---: | ---: |
| 4 | 39.39 | 41.12 | 48.39 |
| 8 | 50.67 | 54.50 | 55.83 |
| **12** | **49.60** | **49.35** | **53.18** |
| 16 | 55.69 | 53.77 | 55.20 |
| 20 | 46.74 | 47.28 | 45.87 |

At layer 12, recall is 0.26% for system, 91.98% for user, 80.69% for reasoning,
69.36% for assistant and 4.48% for tool. Most tool tokens (84.65%) are predicted
as reasoning. Layer 4 has the highest tool recall (55.78%), while layer 16 has
the highest overall token accuracy. Detailed probabilities and
[confusion matrices](APPENDIX.md#confusion-matrices) are in the appendix.

## Message

DepthBench provides much shorter histories than the motivating AppWorld corpus.
Layer 12 exceeds the majority-role baseline and recovers user and reasoning roles
well, but tool and system roles remain poorly recovered. The sanity check therefore
shows a measurable role signal and clear failure patterns to track during training;
it does not establish good overall role recovery or future behavioral improvement.
Despite short individual histories, the 20-attempt corpus contains substantially
more total tokens than ASTRA-SFT-1k.

Fresh-rollout evaluation separately tests task completion and role recovery after
model training. See [RUNNING.md](RUNNING.md) for sources, the failure audit and reproduction.
