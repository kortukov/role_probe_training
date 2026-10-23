# Training GPT-OSS-20B to preserve conversational roles

## Motivation & RQ

Does fine-tuning GPT-OSS-20B with continuously updated role probes make message
roles easier to recover on agent trajectories, while preserving useful behavior?

## Evaluation setup

**Data.** The planned off-policy training corpus contains 300 independently seeded
attempts for each of AppWorld's 90 training tasks: **27,000 trajectories**. Text
will remain fixed during optimization while activations are recomputed as the
model changes. A separate, completed **300-trajectory debug corpus** is reserved
for pipeline correctness, memory and timing checks. Neither corpus is held-out
validation; validation uses separate development tasks. Successful and unsuccessful
attempts remain included.

**Proposed method.** Linear five-class probes predict system, user, reasoning,
assistant and tool roles from pre-MLP activations. Labels come from conversation
metadata. Alternating updates first improve the model's correct-role predictions
with probe weights fixed, then refit the probes on the updated model with model
weights fixed. Probe fitting uses the standard role-confusion C4/Dolma passages
and their original split, separately from the AppWorld model-update corpus. The
objective is to strengthen role recoverability. The starting proposal is a
layer-12 experiment with warm-started probe refits after each model update.

We adapt the continuously updated method in
[Libon et al.](https://arxiv.org/html/2609.38645), replacing its binary behavior
target with correct conversational roles. A KL penalty compares the adapted
model with the unchanged base model on a separate pool of benign prompts and
base-model answers, following the paper's honesty experiment. Layer 12 and correct-role cross-entropy are agreed. Full trajectories are preferred;
the batch size and update budget depend on memory and timing checks.

**Measurements.** Token accuracy is the fraction of labeled content tokens
classified correctly. Macro recall averages recall over the five roles, preventing
common roles from dominating the result. We also report accuracy averaged equally
over trajectories, per-role confusion matrices, and the majority-role baseline.
Validation compares both the continually fitted probes and fresh diagnostic probes;
better agreement with the training probes alone does not establish more generally
recoverable roles. AppWorld task success is measured separately from role accuracy.
Training diagnostics include role loss, KL, gradient norms and time spent on model
updates, activation extraction, probe fitting and validation.

**Throughput calibration.** On two B200 GPUs, we compare processing one, two or
four full trajectories at once per GPU while keeping each optimizer update at
16 trajectories through gradient accumulation. All conditions use the same 80
debug trajectories in five batches, grouping similar lengths within each batch.
The existing layer-12 probe stays fixed. Mean update time uses the four updates
after warmup; peak memory is the largest allocated GPU memory across updates.
Padding fraction counts padding positions divided by all processed positions.
These timings exclude KL and probe refitting and do not measure learning quality.

## Factual observations

The debug corpus is complete: 300 trajectories, all with reasoning, no evaluation
errors, and 117 successful attempts (39%). It contains 4.10 million tokens; median
trajectory length is 12,466 tokens and the maximum is 33,241. The planned
27,000-trajectory training corpus has not been collected; production training has
not started.

| Trajectories per GPU microbatch | Seconds/update | Peak memory/GPU (GiB) | Padding |
|---|---:|---:|---:|
| 1 | 26.6 | 63.0 | 0.0% |
| 2 | 23.2 | 76.5 | 7.9% |
| 4 | 24.5 | 104.1 | 21.9% |

Microbatch 2 achieved 1.15× the throughput of microbatch 1. Microbatch 4 used more
memory and padding without improving on microbatch 2. All settings passed a
separate backward test with the longest trajectory repeated across the microbatch.
A small numerical check found 0.041% loss difference and 0.9989 gradient cosine
similarity between padded and individual forwards; they are close, not identical.
[The appendix](APPENDIX.md) gives numerical checks, loss traces, stress-test memory
and the separate context-length comparison.

## Message

Full trajectories with microbatch 2 are a practical candidate for the next pilot:
they fit comfortably and were fastest in this small comparison. Four timed
updates are not a robust long-run speed estimate; KL and probe-refitting costs
remain unmeasured.

This experiment will test role recoverability on fixed agent text. Whether that
improves behavior on newly generated trajectories requires a separate evaluation.
The debug corpus supports implementation checks, not conclusions about the full
training experiment. Matching trajectory counts alone does not match the paper's
token or optimization budget.

The proposed first training run is described in [DESIGN.md](DESIGN.md).
See [RUNNING.md](RUNNING.md) for collection and reproduction.
