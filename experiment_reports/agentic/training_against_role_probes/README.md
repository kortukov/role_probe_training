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
and their original split, separately from the AppWorld model-update corpus. This strengthens role recoverability. The starting proposal is a
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

## Factual observations

The debug corpus is complete: 300 trajectories, all with reasoning, no evaluation
errors, and 117 successful attempts (39%). It contains 4.10 million tokens; median
trajectory length is 12,466 tokens and the maximum is 33,241. The two-GPU throughput calibration is running on debug data, comparing short
prefixes with full trajectories. Results are pending. The planned 27,000-trajectory
training corpus has not been collected and production training has not started.

## Message

This experiment will test role recoverability on fixed agent text. Whether that
improves behavior on newly generated trajectories requires a separate evaluation.
The debug corpus supports implementation checks, not conclusions about the full
training experiment. Matching trajectory counts alone does not match the paper's
token or optimization budget.

The proposed first training run is described in [DESIGN.md](DESIGN.md).
See [RUNNING.md](RUNNING.md) for collection and reproduction.
