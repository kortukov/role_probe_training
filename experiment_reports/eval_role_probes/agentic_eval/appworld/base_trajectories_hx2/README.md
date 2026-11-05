# GPT-OSS-20B on AppWorld normal test: hx2 base cohort

Status: complete. Date: 2026-10-30.

## 1. Motivation & RQ

The [probe training-data comparison](../../../../gptoss_role_probes/training_data/README.md) needs
whole agent histories from the unchanged model. The coauthor's AppWorld trajectories
cannot be copied to hx2, so this experiment regenerates the normal-test split here
with the same collection code and settings. It is a new sample, not a reproduction
of the [original collection](../base_trajectories/README.md).

## 2. Evaluation setup

All 168 AppWorld normal-test tasks, one attempt each, with the official container
(AppWorld 0.1.3, image SHA-256 recorded in `metrics/test_normal_manifest.json`) and
data bundle 0.1.0. GPT-OSS-20B is served natively (MXFP4 experts) by vLLM 0.32.0 on
one H200 with high reasoning, temperature 1, seed 123, at most 50 Python
interactions, 8,192 generated tokens per call, a 32,768-token context, and 1,800
seconds per task; eight worlds run concurrently. Task success is the fraction of
assigned tasks passing AppWorld's state-based evaluation, with errors counted as
failures. A four-task training pilot at concurrency 1 and 8 preceded the run.

## 3. Factual observations

| Tasks solved | Task success | 95% Wilson interval | Sample errors | Limit stops | Wall time |
|---|---:|---|---:|---:|---:|
| 53 / 168 | 31.5% | 24.9%–39.0% | 0 | 0 | 21.7 min |

All 168 histories exported with reasoning; 167 contain tool replies, and one ends
after reasoning with an empty answer. Histories span 5,543 to 32,256 tokens in
retrospective Harmony, so every one is scorable whole. The original cohort on the
coauthor's cluster solved 26.8%; the difference is within sampling noise at this size
and the two cohorts are not compared further.

## 4. Message

The run provides a complete, error-free set of tool-bearing base-model histories
for probe evaluation on hx2. Its task score is a second sample of the same
configuration and should be reported separately from the published cohort.

[How to run this experiment](RUNNING.md)
