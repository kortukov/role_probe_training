# Role recovery on DepthBench

## Motivation & RQ

Do existing role probes recover conversational roles on short GPT-OSS-20B tool-use
histories before model training? DepthBench is the dataset/environment released
with TIER; the local experiment uses the dataset name.

## Evaluation setup

**Data.** Collected 20 independently seeded rollouts for every one of the 944
upstream training prompts, retaining separate required-call groups 0–6. Explicitly
combine groups 1, 2 and 3 into the model-update corpus. The totals are
18,880 collected rollouts and 12,980 model-update histories. Required-call count
comes from the task definition; actual agent calls may differ. No validation or
test tasks enter collection. Each task exposes relevant tools and three seeded
distractors, using local Python implementations and fresh state per rollout.

**Probes.** Fixed existing five-class linear probes read GPT-OSS-20B pre-MLP
activations at layers 4, 8, 12, 16 and 20; layer 12 is primary. Ground truth comes
from message/channel metadata. No probe is fitted on this corpus. Evaluation
preserves complete retrospective Harmony context and excludes structural tokens
and synthetic tool-schema context.

**Metrics and baselines.** Token accuracy weights labeled content tokens equally;
macro recall averages per-role recall; mean trajectory accuracy weights histories
equally. The majority-role baseline predicts the most common content role.
Per-role recall and correct-label probability are different measurements.
This training-corpus sanity check does not measure held-out transfer or task success.

## Factual observations

The pinned upstream split yields 50/410/55/184/145/60/40 prompts for required-call
counts 0/1/2/3/4/5/6. Exact task/attempt coverage has been verified for all18,880
histories. The combined1–3 corpus contains12,980 histories, all with reasoning;
12,418 also contain tool responses. One training attempt ended after a malformed
generated tool name caused the next API request to be rejected. Its original
partial history and error are retained unchanged. A separate group5 attempt hit
the output-token cap. No attempts were discarded or regenerated based on success.
The full training-corpus probe evaluation is running or queued; no probe results
are reported yet.

## Message

The corpus design separates collection groups from the explicit 1–3 training
mixture. Whether it exhibits recoverable roles remains to be measured before
training. Production training also requires the shared implementation's READY signal.

See [RUNNING.md](RUNNING.md) for pinned sources and reproduction.
