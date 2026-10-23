# Supplementary calibration results

## Context length

Two B200s process an effective batch of 16 trajectories, one trajectory per GPU
microbatch. The table reports mean wall time over two updates after one warmup,
taking the slower rank per update. Peak memory is the maximum allocated memory
on either GPU across the updates. The probe is fixed; KL and refitting are excluded.
Different contexts retain prefixes of the same examples. This small context run
is separate from the paired microbatch comparison in the main report.

| Context | Seconds/update | Peak update memory/GPU (GiB) |
|---|---:|---:|
| First 4,096 tokens | 11.8 | 51.8 |
| First 8,192 tokens | 16.9 | 53.7 |
| First 16,384 tokens | 20.2 | 57.4 |
| Full trajectories | 21.5 | 59.9 |

The longest full trajectory has 33,241 tokens; its separate backward test peaked
at 63.8 GiB. A single pass over 27,000 trajectories at batch 16 needs 1,688 updates.
Extrapolating the original full-context mean gives 10.1 hours for role updates
only, conditional on similar trajectory lengths and throughput. This excludes
KL, probe refitting, validation, checkpointing and trajectory generation.

Source: [context_results.json](context_results.json).

## Microbatch memory stress test

The stress test repeats the same 33,241-token trajectory across every slot of the
microbatch on each GPU. Peak allocated memory includes the already initialized
optimizer state, but the stress test itself is a backward pass, not another update.

| Microbatch/GPU | Peak stress memory/GPU (GiB) | Measured update range (s) |
|---|---:|---:|
| 1 | 63.8 | 25.6–28.0 |
| 2 | 79.7 | 21.3–24.3 |
| 4 | 111.8 | 20.8–27.0 |

## Short training-loss traces

These are token-weighted training cross-entropies before each optimizer update,
including the warmup update. Each column uses the same trajectory batch across
microbatch settings. Different columns contain different trajectories; the
numbers are diagnostics, not held-out performance or convergence evidence.

| Microbatch/GPU | Update 1 | Update 2 | Update 3 | Update 4 | Update 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 3.0027 | 2.1007 | 1.2391 | 0.8969 | 0.7424 |
| 2 | 3.0027 | 2.0997 | 1.2388 | 0.8970 | 0.7446 |
| 4 | 3.0028 | 2.0994 | 1.2378 | 0.8967 | 0.7448 |

Source: [microbatch_results.json](microbatch_results.json).
