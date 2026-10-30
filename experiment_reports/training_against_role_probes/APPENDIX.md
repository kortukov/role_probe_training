# Supplementary results

Rows denote true metadata-derived roles;
columns denote predicted roles. Values are row-normalized percentages.

## Training-corpus baseline confusions

Rows are ground-truth content roles; columns are predicted roles. Metrics include
all metadata-labeled content tokens from the 2,700 full trajectories.

### Layer 4


Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 7.0% | 25.4% | 4.5% | 27.5% | 35.7% |
| user | 0.3% | 50.7% | 9.7% | 25.6% | 13.7% |
| cot | 0.1% | 2.8% | 49.6% | 5.7% | 41.9% |
| assistant | 0.1% | 0.2% | 10.2% | 2.3% | 87.2% |
| tool | 0.1% | 0.1% | 7.1% | 4.0% | 88.6% |

### Layer 8


Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.6% | 15.6% | 16.8% | 0.0% | 61.1% |
| user | 0.0% | 92.7% | 1.7% | 0.0% | 5.5% |
| cot | 0.1% | 2.5% | 91.7% | 2.3% | 3.5% |
| assistant | 0.3% | 0.5% | 86.7% | 8.5% | 4.0% |
| tool | 14.8% | 25.2% | 32.5% | 7.0% | 20.5% |

### Layer 12


Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.2% | 0.8% | 7.4% | 0.0% | 90.6% |
| user | 0.0% | 44.4% | 1.5% | 0.0% | 54.1% |
| cot | 0.0% | 4.1% | 74.6% | 13.1% | 8.3% |
| assistant | 0.1% | 0.7% | 63.1% | 33.7% | 2.3% |
| tool | 7.9% | 16.6% | 41.4% | 19.4% | 14.7% |

### Layer 16


Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.5% | 28.7% | 4.1% | 0.0% | 64.8% |
| user | 0.2% | 61.0% | 0.9% | 0.0% | 37.9% |
| cot | 0.6% | 0.4% | 15.8% | 6.1% | 77.1% |
| assistant | 2.4% | 0.1% | 18.6% | 22.6% | 56.3% |
| tool | 12.4% | 14.9% | 3.0% | 24.5% | 45.2% |

### Layer 20


Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.8% | 54.9% | 10.7% | 0.0% | 33.6% |
| user | 0.1% | 88.4% | 1.1% | 0.2% | 10.2% |
| cot | 0.8% | 7.3% | 23.3% | 5.1% | 63.4% |
| assistant | 1.9% | 8.7% | 21.8% | 11.4% | 56.1% |
| tool | 9.0% | 31.2% | 3.2% | 8.4% | 48.2% |

<!-- development-results -->
## Development confusions

Rows are true content roles; columns are predicted roles. Values are row-normalized percentages.

### Before training


Row-normalized percentages:

| True / predicted | system | user | reasoning | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.2% | 0.8% | 7.4% | 0.0% | 90.6% |
| user | 0.0% | 41.1% | 1.7% | 0.0% | 57.3% |
| reasoning | 0.0% | 6.7% | 77.4% | 6.8% | 9.1% |
| assistant | 0.1% | 0.7% | 56.5% | 40.6% | 2.2% |
| tool | 6.4% | 17.8% | 42.1% | 16.9% | 16.8% |

### After update 169


Row-normalized percentages:

| True / predicted | system | user | reasoning | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| user | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| reasoning | 0.0% | 0.2% | 99.8% | 0.0% | 0.0% |
| assistant | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| tool | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
<!-- /development-results -->

## Retail length and generation failures

Saved histories retain every reasoning message. The native serving formatter omits
historical reasoning, so concatenated replay length is not the input length of any
single generation call. The largest recorded API inputs were 8,996 tokens for the
base model, 15,921 for AppWorld-trained, 32,646 for ASTRA-trained and 31,784 for
DepthBench-trained, all within the 32,768-token serving context.

| Measure (114 tasks per model) | Base | AppWorld-trained | ASTRA-trained | DepthBench-trained |
|---|---:|---:|---:|---:|
| Tasks solved | 73 | 27 | 44 | 43 |
| Median reconstructed-history tokens | 23,382 | 90,388 | 40,913 | 78,575 |
| Histories above 131,072 tokens | 1 | 38 | 11 | 19 |
| Successful tasks among those long histories | 0 | 4 | 4 | 4 |
| Histories with at least one output-cap stop | 64 | 111 | 80 | 100 |
| Reasoning-only assistant turns | 96 / 1,580 | 631 / 1,678 | 280 / 1,661 | 455 / 1,646 |

An output-cap stop ends a generation, not automatically the retail task. The
simulator removes reasoning from the assistant reply; a reasoning-only response
therefore supplies no visible answer and the customer can continue asking for
help. The loop ends at a simulator stop marker or an overall execution limit.
The task scorer checks the required database changes and communicated information;
`error = None` means no evaluation exception, not task success.

Inspection found degenerate repetition, not only longer coherent planning. On
retail task 37, the AppWorld-trained model repeated “The user is basically asking
for confirmation” 2,034 times in one reasoning message and failed a task the base
model solved. On task 110, the DepthBench-trained model repeated “The assistant
responded with nothing” 1,623 times in one reasoning message, made no tool calls,
and failed where the base model succeeded. These are inspected examples, not an
exhaustive automatic classification of incoherence. Four long histories in each
trained cohort still succeeded, so overlength alone must not be relabeled failure.

The old base run and trained runs differ in serving precision/software; this
comparison documents observed failure patterns and does not isolate their cause.
[Machine-readable length audit](post_training/retail_length_audit.json) and
[the audit script](../../src/experiments/training_against_role_probes/post_training/audit_retail.py)
record lengths, caps, and task outcomes. Native rendering and simulator stopping
rules were checked in the pinned installed runtime source.

## Fresh-probe fitting check

The three trained models each receive a new multinomial linear layer-12 probe,
fitted from scratch on the original 250 C4/Dolma passages. All content tokens,
the original passage split, C=0.005 and the pre-MLP readout are retained. This is
independent of the warm-started probes used during model training. No benchmark
trajectories are used for fitting or selecting probes.

| Model | Accuracy on held-out standard passages |
|---|---:|
| Base (original probe) | 68.93% |
| AppWorld-trained (fresh probe) | 91.06% |
| ASTRA-trained (fresh probe) | 87.86% |
| DepthBench-trained (fresh probe) | 82.44% |

These are controlled-text probe checks, not agent-task scores. AppWorld and
DepthBench fresh fits emitted line-search warnings; no strict convergence claim
is made. Weights, exact source/model hashes and fitting diagnostics accompany the
[AppWorld](post_training/probes/layer_12.json),
[ASTRA](astra_sft_1k/post_training/probes/layer_12.json), and
[DepthBench](depthbench/post_training/probes/layer_12.json) artifacts.

<!-- fresh-rollout-confusions -->
## Fresh-trajectory confusions

Rows are true roles; columns are predicted roles. Each row sums to 100% (up to rounding). Only whole histories within native context are scored. Over-limit histories retain their task outcome but have no role score.


### AppWorld test_normal — Base

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 1.23% | 0.82% | 7.38% | 0.00% | 90.57% |
| User | 0.02% | 49.94% | 1.94% | 0.00% | 48.10% |
| Reasoning | 0.01% | 3.43% | 73.14% | 16.54% | 6.88% |
| Assistant | 0.17% | 0.65% | 61.69% | 35.39% | 2.09% |
| Tool | 9.21% | 13.25% | 41.38% | 22.84% | 13.32% |

### AppWorld test_normal — AppWorld-trained

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| User | 0.06% | 99.57% | 0.36% | 0.00% | 0.01% |
| Reasoning | 0.00% | 0.03% | 99.95% | 0.00% | 0.01% |
| Assistant | 0.00% | 0.00% | 1.45% | 91.02% | 7.53% |
| Tool | 0.00% | 0.01% | 0.00% | 0.00% | 99.99% |

### AppWorld test_normal — ASTRA-trained

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 47.54% | 2.05% | 20.49% | 0.00% | 29.92% |
| User | 0.15% | 89.14% | 1.34% | 0.00% | 9.37% |
| Reasoning | 0.00% | 0.42% | 93.94% | 5.63% | 0.01% |
| Assistant | 0.00% | 0.05% | 77.45% | 21.83% | 0.66% |
| Tool | 0.04% | 0.02% | 1.90% | 0.07% | 97.97% |

### AppWorld test_normal — DepthBench-trained

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 48.77% | 0.00% | 24.18% | 0.00% | 27.06% |
| User | 0.31% | 79.02% | 0.38% | 0.00% | 20.29% |
| Reasoning | 0.00% | 0.41% | 97.81% | 1.72% | 0.07% |
| Assistant | 0.00% | 0.00% | 34.86% | 63.02% | 2.11% |
| Tool | 0.45% | 7.97% | 5.04% | 0.79% | 85.75% |

### τ² retail — Base

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 6.64% | 35.55% | 41.58% | 0.00% | 16.23% |
| User | 2.37% | 29.89% | 11.40% | 56.33% | 0.01% |
| Reasoning | 0.67% | 0.24% | 20.03% | 79.05% | 0.01% |
| Assistant | 0.05% | 0.00% | 4.98% | 94.95% | 0.02% |
| Tool | 1.24% | 0.08% | 24.23% | 73.76% | 0.69% |

### τ² retail — AppWorld-trained

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 49.90% | 21.80% | 28.30% | 0.00% | 0.00% |
| User | 24.70% | 32.14% | 7.89% | 30.04% | 5.24% |
| Reasoning | 25.82% | 0.01% | 52.29% | 21.87% | 0.01% |
| Assistant | 0.65% | 0.00% | 0.08% | 95.90% | 3.37% |
| Tool | 0.03% | 0.00% | 0.14% | 0.17% | 99.66% |

### τ² retail — ASTRA-trained

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 28.17% | 22.87% | 23.68% | 0.00% | 25.29% |
| User | 12.94% | 39.42% | 3.66% | 43.95% | 0.03% |
| Reasoning | 5.92% | 0.02% | 38.00% | 56.02% | 0.04% |
| Assistant | 2.92% | 0.01% | 11.95% | 83.49% | 1.64% |
| Tool | 0.09% | 0.00% | 4.26% | 2.63% | 93.02% |

### τ² retail — DepthBench-trained

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 65.46% | 16.57% | 14.29% | 0.00% | 3.69% |
| User | 19.89% | 37.56% | 9.53% | 32.95% | 0.06% |
| Reasoning | 4.14% | 0.01% | 40.58% | 55.21% | 0.06% |
| Assistant | 0.05% | 0.00% | 0.28% | 99.63% | 0.03% |
| Tool | 1.26% | 0.16% | 0.43% | 0.49% | 97.67% |

<!-- /fresh-rollout-confusions -->

<!-- whole-history-coverage -->
## Whole-history coverage

| Benchmark | Model | Histories scored | Unscorable | Evaluation errors |
|---|---|---:|---:|---:|
| AppWorld test_normal | Base | 168/168 | 0 | 0 |
| AppWorld test_normal | AppWorld-trained | 168/168 | 0 | 0 |
| AppWorld test_normal | ASTRA-trained | 168/168 | 0 | 0 |
| AppWorld test_normal | DepthBench-trained | 168/168 | 0 | 0 |
| τ² retail | Base | 113/114 | 1 | 0 |
| τ² retail | AppWorld-trained | 76/114 | 38 | 0 |
| τ² retail | ASTRA-trained | 103/114 | 11 | 0 |
| τ² retail | DepthBench-trained | 95/114 | 19 | 0 |
| SWE-bench Verified | Base | Pending | Pending | 2 |
| SWE-bench Verified | AppWorld-trained | 500/500 | 0 | 0 |
| SWE-bench Verified | DepthBench-trained | 498/500 | 2 | 0 |
| SWE-bench Verified | AppWorld + agentic KL (800) | Pending | Pending | 0 |
| SWE-bench Verified | AppWorld + same-batch KL (2,700) | Pending | Pending | 0 |
<!-- /whole-history-coverage -->

## SWE-bench whole-history results

Task scores use all 500 attempts per model. Role metrics use complete conversations within native context, with no chunking or truncation. Base, agentic-KL, and same-batch KL role metrics are pending. Rows below are true roles, columns are predicted roles; entries are row-normalized percentages.

### AppWorld-trained

Coverage: 500/500 histories; 0 excluded above native context. Token-weighted accuracy: 87.55%. Mean per-conversation accuracy: 87.08%.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 39.8 | 50.0 | 9.7 | 0.0 | 0.5 |
| user | 0.2 | 93.8 | 1.5 | 1.5 | 3.0 |
| cot | 0.0 | 1.7 | 98.0 | 0.1 | 0.2 |
| assistant | 0.0 | 0.0 | 1.6 | 90.6 | 7.8 |
| tool | 2.2 | 10.1 | 2.2 | 0.0 | 85.5 |

[Validated metrics and provenance](swe_generous/appworld_role_results.json).

### DepthBench-trained

Coverage: 498/500 histories; 2 excluded above native context. Token-weighted accuracy: 48.36%. Mean per-conversation accuracy: 50.22%.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 60.6 | 5.6 | 18.1 | 0.0 | 15.7 |
| user | 1.2 | 89.2 | 4.4 | 2.7 | 2.6 |
| cot | 0.0 | 0.7 | 97.3 | 1.8 | 0.2 |
| assistant | 0.0 | 0.0 | 47.8 | 49.6 | 2.6 |
| tool | 7.1 | 28.3 | 30.1 | 4.8 | 29.7 |

[Validated metrics and provenance](swe_generous/depthbench_role_results.json).
