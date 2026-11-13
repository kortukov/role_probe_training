# Off-policy training on 2,700 rollouts under the unified setup

Status: prepared, not run (2026-11-13). Run 1 of the [training status page](../README.md).

## Motivation & RQ

The best off-policy recipe so far (same-batch KL on fixed base-model rollouts) was found
with the old chat templates and layer-12 cuML probes. Does it still raise role accuracy
while keeping task success when rollouts come from the Harmony renderer and the probe has ten
roles (one per Harmony header form), sits at layer 16 and is refitted to convergence after
every update?

## Evaluation setup

**Training.** GPT-OSS-20B with rank-64 LoRA adapters, one epoch over 2,700 base-model
AppWorld rollouts (30 attempts on each of the 90 training tasks, no filtering by outcome),
169 updates of 16 rollouts. Each update adds the cross-entropy of the ten-role probe and,
with coefficient 16, the KL divergence from the base model on the tokens the model wrote in
the same 16 rollouts. After every update the probe is refitted to convergence on the 250
neutral passages, each wrapped in all ten headers. Whole histories, no truncation.

**Evaluation.** A probe fitted from scratch on the trained model, scored on fresh runs of
AppWorld test, τ² retail and SWE-bench at temperature 1.

**Metrics and baseline.** Task success over all assigned tasks, and token-level role
accuracy overall and per role. Baseline: the unchanged model under the same setup
([stage 1 status](../../eval_role_probes/README.md)). Old-template counterpart:
"same-batch KL (2,700)" in [the old results](../OLD_TEMPLATE_RESULTS.md).

## Factual observations

None yet.

## Message

None yet.

[Reproduction instructions](RUNNING.md)
