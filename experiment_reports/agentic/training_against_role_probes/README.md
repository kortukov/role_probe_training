# Training GPT-OSS-20B to preserve conversational roles

## Motivation & RQ

Does fine-tuning GPT-OSS-20B with continuously updated role probes make message
roles easier to recover on agent trajectories, while preserving useful behavior?

## Evaluation setup

**Data.** We collect 300 fresh AppWorld training attempts once and keep their text
fixed during optimization. Three independently seeded passes cover all 90 training
tasks; a fourth covers 30 seeded tasks. Thus each task contributes three or four
attempts. Successful and unsuccessful attempts are retained. This is off-policy
training: as the model changes, we recompute activations on the original text,
without generating replacement trajectories. Separate development tasks will
provide validation; repeated attempts at a training task are not held-out data.

**Proposed method.** Linear five-class probes predict system, user, reasoning,
assistant and tool roles from pre-MLP activations. Labels come from conversation
metadata. Alternating updates first improve the model's correct-role predictions
with probe weights fixed, then refit the probes on the updated model with model
weights fixed. This strengthens role recoverability. The starting proposal is a
layer-12 experiment with warm-started probe refits after each model update.

We adapt the continuously updated method in
[Libon et al.](https://arxiv.org/html/2609.38645), replacing its binary behavior
target with correct conversational roles. A KL penalty compares the adapted
model with the unchanged base model on a separate pool of benign prompts and
base-model answers, following the paper's honesty experiment. The role loss,
sequence length and update budget remain proposals pending discussion.

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

Four collection shards are running under Slurm array 1492660, with automatic
corpus validation queued as job 1492664. No model optimization or probe refitting
has been performed in this experiment, and no training outcome is available yet.

## Message

This experiment will test role recoverability on fixed agent text. Whether that
improves behavior on newly generated trajectories requires a separate evaluation.
Three hundred trajectories are a starting corpus size, not an established match
to the paper's data budget; token and sequence counts will determine its scale.

The proposed first training run is described in [DESIGN.md](DESIGN.md).
See [RUNNING.md](RUNNING.md) for collection and reproduction.
