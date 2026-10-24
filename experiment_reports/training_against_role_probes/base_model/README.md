# Whole-history base-model reference

## Motivation & RQ

Provide the unchanged GPT-OSS-20B reference for the [trained-model comparison](../README.md).

## Evaluation setup

Reuse the existing AppWorld normal-test, SWE-bench Verified and τ² retail
trajectories and task scores. Recompute role predictions with the original
standard-data probes at layers 4, 8, 12, 16 and 20, using one complete conversation
per forward pass. No new model generations or probe fits are needed. The one
retail history above 131,072 tokens is unscorable; its task outcome remains included.
Role accuracy is the fraction of content tokens classified correctly; per-role
accuracy is recall. Task success counts all selected tasks, including errors.

## Factual observations

Results are consolidated in the [main comparison](../README.md) and the
[whole-history layer comparison](../../agentic/probe_layer_comparison/README.md).
The earlier SWE task run contains 226 API-error attempts and different decoding
settings; its task score must not be treated as a controlled training comparison.

## Message

These measurements reuse the original benchmark attempts while removing artificial
context resets from probe evaluation. Whole replay retains all saved reasoning;
it is not the exact generation-time prompt, which can omit historical reasoning.

[Reproduction instructions](RUNNING.md)
