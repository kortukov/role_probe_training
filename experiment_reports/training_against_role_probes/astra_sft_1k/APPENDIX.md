# Baseline supporting results

All 1,000 training histories were evaluated before model updates. Roles come from message/channel metadata. `cot` denotes assistant reasoning. Only original content tokens contribute; headers, boundaries and synthetic tool schemas are excluded.

## Context lengths

These lengths include the complete rendered context, including tool schemas and structural tokens; they differ from the scored content-token denominator.

| Total tokens | Median history | 95th percentile | Longest history |
| ---: | ---: | ---: | ---: |
| 4,178,050 | 3,724 | 7,906 | 15,439 |

## Per-role results

Recall is the fraction of a role’s content tokens whose highest-probability label is correct. Mean correct-label probability averages the probability assigned to the ground-truth role; it is not argmax accuracy. Both are percentages.

| Layer | Role | Content tokens | Recall (%) | Mean correct-label probability (%) |
| ---: | --- | ---: | ---: | ---: |
| 4 | system | 23,000 | 15.94 | 17.30 |
| 4 | user | 84,116 | 31.70 | 25.06 |
| 4 | cot | 468,388 | 61.98 | 48.93 |
| 4 | assistant | 756,922 | 26.60 | 28.99 |
| 4 | tool | 1,444,546 | 68.17 | 52.70 |
| 8 | system | 23,000 | 26.31 | 26.76 |
| 8 | user | 84,116 | 91.38 | 85.83 |
| 8 | cot | 468,388 | 79.43 | 76.33 |
| 8 | assistant | 756,922 | 84.03 | 83.78 |
| 8 | tool | 1,444,546 | 21.55 | 21.69 |
| 12 | system | 23,000 | 6.55 | 8.34 |
| 12 | user | 84,116 | 79.85 | 74.00 |
| 12 | cot | 468,388 | 59.02 | 55.79 |
| 12 | assistant | 756,922 | 91.25 | 90.38 |
| 12 | tool | 1,444,546 | 10.69 | 12.82 |
| 16 | system | 23,000 | 31.93 | 32.21 |
| 16 | user | 84,116 | 85.83 | 82.23 |
| 16 | cot | 468,388 | 50.74 | 48.64 |
| 16 | assistant | 756,922 | 95.70 | 94.87 |
| 16 | tool | 1,444,546 | 8.39 | 9.14 |
| 20 | system | 23,000 | 10.21 | 13.22 |
| 20 | user | 84,116 | 95.81 | 93.27 |
| 20 | cot | 468,388 | 57.46 | 55.60 |
| 20 | assistant | 756,922 | 89.75 | 89.01 |
| 20 | tool | 1,444,546 | 14.89 | 15.60 |

## Confusion matrices

Rows are ground-truth roles and columns are predicted roles. Cells are content-token counts; row-normalized percentages follow each count matrix. No probe fitting uses these histories.

### Layer 4


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 15.94 | 3.54 | 6.90 | 4.89 | 68.74 |
| user | 1.38 | 31.70 | 13.95 | 21.57 | 31.39 |
| cot | 0.08 | 1.31 | 61.98 | 14.35 | 22.29 |
| assistant | 0.12 | 0.27 | 20.72 | 26.60 | 52.29 |
| tool | 0.68 | 0.93 | 17.87 | 12.35 | 68.17 |

### Layer 8


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 26.31 | 0.39 | 9.42 | 0.00 | 63.88 |
| user | 2.64 | 91.38 | 2.10 | 0.19 | 3.69 |
| cot | 0.02 | 1.77 | 79.43 | 16.81 | 1.97 |
| assistant | 0.19 | 0.29 | 14.92 | 84.03 | 0.57 |
| tool | 3.08 | 4.86 | 39.67 | 30.84 | 21.55 |

### Layer 12


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 6.55 | 0.00 | 56.91 | 0.00 | 36.53 |
| user | 0.22 | 79.85 | 12.29 | 0.24 | 7.41 |
| cot | 0.02 | 1.39 | 59.02 | 38.29 | 1.27 |
| assistant | 0.21 | 0.36 | 7.59 | 91.25 | 0.59 |
| tool | 2.73 | 4.02 | 38.77 | 43.78 | 10.69 |

### Layer 16


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 31.93 | 2.45 | 8.42 | 1.76 | 55.44 |
| user | 1.99 | 85.83 | 3.74 | 0.92 | 7.53 |
| cot | 0.12 | 0.12 | 50.74 | 33.42 | 15.61 |
| assistant | 0.25 | 0.20 | 1.35 | 95.70 | 2.50 |
| tool | 1.68 | 5.13 | 6.49 | 78.32 | 8.39 |

### Layer 20


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 10.21 | 5.23 | 21.12 | 0.13 | 63.31 |
| user | 0.42 | 95.81 | 0.44 | 0.35 | 2.98 |
| cot | 0.15 | 4.91 | 57.46 | 26.74 | 10.74 |
| assistant | 0.24 | 6.24 | 1.48 | 89.75 | 2.29 |
| tool | 2.40 | 34.27 | 5.59 | 42.86 | 14.89 |

Machine-readable results: [baseline_metrics.json](baseline_metrics.json). Execution metadata: [baseline_manifest.json](baseline_manifest.json).

<!-- fresh-rollout-confusions -->
## Fresh-trajectory confusions

Each row is a ground-truth content role and each column a predicted role. Values are percentages of tokens in that row; rounding may affect totals. A dash denotes a role absent from the cohort. Probe scores include only whole histories within native context; task scores include every attempt.


### AppWorld test_normal

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 47.54% | 2.05% | 20.49% | 0.00% | 29.92% |
| User | 0.15% | 89.14% | 1.34% | 0.00% | 9.37% |
| Reasoning | 0.00% | 0.42% | 93.94% | 5.63% | 0.01% |
| Assistant | 0.00% | 0.05% | 77.45% | 21.83% | 0.66% |
| Tool | 0.04% | 0.02% | 1.90% | 0.07% | 97.97% |

Probe coverage: 168/168 whole histories. 0 exceed native context and have no probe score; task completion still includes every attempt.

### τ² retail

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 28.17% | 22.87% | 23.68% | 0.00% | 25.29% |
| User | 12.94% | 39.42% | 3.66% | 43.95% | 0.03% |
| Reasoning | 5.92% | 0.02% | 38.00% | 56.02% | 0.04% |
| Assistant | 2.92% | 0.01% | 11.95% | 83.49% | 1.64% |
| Tool | 0.09% | 0.00% | 4.26% | 2.63% | 93.02% |

Probe coverage: 103/114 whole histories. 11 exceed native context and have no probe score; task completion still includes every attempt.

### SWE-bench Verified

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 18.98% | 16.20% | 3.24% | 0.00% | 61.57% |
| User | 0.59% | 87.46% | 3.12% | 3.06% | 5.77% |
| Reasoning | 0.04% | 0.43% | 93.98% | 5.53% | 0.02% |
| Assistant | 0.05% | 0.03% | 65.88% | 33.16% | 0.88% |
| Tool | 8.15% | 1.09% | 17.77% | 3.45% | 69.54% |

Probe coverage: 500/500 whole histories. 0 exceed native context and have no probe score; task completion still includes every attempt.
<!-- /fresh-rollout-confusions -->
