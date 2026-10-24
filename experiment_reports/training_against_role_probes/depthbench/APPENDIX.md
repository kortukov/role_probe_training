# Baseline supporting results

All 12,980 training histories were evaluated before model updates. Roles come from message/channel metadata. `cot` denotes assistant reasoning. Only original content tokens contribute; headers, boundaries and synthetic tool schemas are excluded.

## Context lengths

These lengths include the complete rendered context, including tool schemas and structural tokens; they differ from the scored content-token denominator.

| Total tokens | Median history | 95th percentile | Longest history |
| ---: | ---: | ---: | ---: |
| 22,132,417 | 1,087 | 4,589 | 8,426 |

## Per-role results

Recall is the fraction of a role’s content tokens whose highest-probability label is correct. Mean correct-label probability averages the probability assigned to the ground-truth role; it is not argmax accuracy. Both are percentages.

| Layer | Role | Content tokens | Recall (%) | Mean correct-label probability (%) |
| ---: | --- | ---: | ---: | ---: |
| 4 | system | 649,000 | 34.33 | 27.80 |
| 4 | user | 208,020 | 49.07 | 34.96 |
| 4 | cot | 3,670,445 | 52.44 | 43.96 |
| 4 | assistant | 4,365,048 | 13.96 | 19.31 |
| 4 | tool | 3,927,628 | 55.78 | 46.25 |
| 8 | system | 649,000 | 15.37 | 19.43 |
| 8 | user | 208,020 | 99.68 | 98.98 |
| 8 | cot | 3,670,445 | 95.72 | 91.82 |
| 8 | assistant | 4,365,048 | 57.47 | 58.48 |
| 8 | tool | 3,927,628 | 4.26 | 8.89 |
| 12 | system | 649,000 | 0.26 | 1.74 |
| 12 | user | 208,020 | 91.98 | 87.35 |
| 12 | cot | 3,670,445 | 80.69 | 75.96 |
| 12 | assistant | 4,365,048 | 69.36 | 70.42 |
| 12 | tool | 3,927,628 | 4.48 | 8.43 |
| 16 | system | 649,000 | 11.15 | 13.61 |
| 16 | user | 208,020 | 87.38 | 84.74 |
| 16 | cot | 3,670,445 | 70.56 | 66.35 |
| 16 | assistant | 4,365,048 | 86.24 | 83.69 |
| 16 | tool | 3,927,628 | 13.53 | 14.74 |
| 20 | system | 649,000 | 2.50 | 5.09 |
| 20 | user | 208,020 | 91.42 | 87.04 |
| 20 | cot | 3,670,445 | 54.70 | 52.84 |
| 20 | assistant | 4,365,048 | 75.57 | 74.18 |
| 20 | tool | 3,927,628 | 12.21 | 12.64 |

## Confusion matrices

Rows are ground-truth roles and columns are predicted roles. Cells are content-token counts; row-normalized percentages follow each count matrix. No probe fitting uses these histories.

### Layer 4


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 34.33 | 10.76 | 0.66 | 2.06 | 52.19 |
| user | 1.77 | 49.07 | 16.97 | 18.74 | 13.45 |
| cot | 0.07 | 1.01 | 52.44 | 4.96 | 41.53 |
| assistant | 0.03 | 0.08 | 23.83 | 13.96 | 62.10 |
| tool | 0.03 | 0.11 | 37.06 | 7.01 | 55.78 |

### Layer 8


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 15.37 | 22.84 | 18.89 | 0.00 | 42.90 |
| user | 0.05 | 99.68 | 0.13 | 0.12 | 0.02 |
| cot | 0.01 | 1.72 | 95.72 | 1.50 | 1.05 |
| assistant | 0.08 | 0.20 | 41.53 | 57.47 | 0.72 |
| tool | 0.12 | 0.50 | 92.22 | 2.91 | 4.26 |

### Layer 12


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 0.26 | 6.10 | 38.03 | 0.00 | 55.61 |
| user | 0.01 | 91.98 | 7.02 | 0.06 | 0.93 |
| cot | 0.00 | 1.14 | 80.69 | 17.46 | 0.70 |
| assistant | 0.11 | 0.15 | 29.80 | 69.36 | 0.59 |
| tool | 0.15 | 1.55 | 84.65 | 9.17 | 4.48 |

### Layer 16


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 11.15 | 35.72 | 1.33 | 0.00 | 51.81 |
| user | 0.57 | 87.38 | 2.35 | 2.25 | 7.45 |
| cot | 0.45 | 0.47 | 70.56 | 4.81 | 23.71 |
| assistant | 0.56 | 0.26 | 6.75 | 86.24 | 6.19 |
| tool | 1.93 | 0.67 | 15.84 | 68.04 | 13.53 |

### Layer 20


| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 2.50 | 40.96 | 8.87 | 0.15 | 47.51 |
| user | 0.47 | 91.42 | 1.23 | 0.92 | 5.96 |
| cot | 0.57 | 8.64 | 54.70 | 5.29 | 30.80 |
| assistant | 0.93 | 11.24 | 7.46 | 75.57 | 4.80 |
| tool | 1.74 | 3.69 | 10.05 | 72.31 | 12.21 |

Machine-readable results: [baseline_metrics.json](baseline_metrics.json). Execution metadata: [baseline_manifest.json](baseline_manifest.json).

<!-- fresh-rollout-confusions -->
## Fresh-trajectory confusions

Each row is a ground-truth content role and each column a predicted role. Values are percentages of tokens in that row; rounding may affect totals. A dash denotes a role absent from the cohort. Probe scores include only whole histories within native context; task scores include every attempt.


### AppWorld test_normal

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 48.77% | 0.00% | 24.18% | 0.00% | 27.06% |
| User | 0.31% | 79.02% | 0.38% | 0.00% | 20.29% |
| Reasoning | 0.00% | 0.41% | 97.81% | 1.72% | 0.07% |
| Assistant | 0.00% | 0.00% | 34.86% | 63.02% | 2.11% |
| Tool | 0.45% | 7.97% | 5.04% | 0.79% | 85.75% |

Probe coverage: 168/168 whole histories. 0 exceed native context and have no probe score; task completion still includes every attempt.

### τ² retail

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 65.46% | 16.57% | 14.29% | 0.00% | 3.69% |
| User | 19.89% | 37.56% | 9.53% | 32.95% | 0.06% |
| Reasoning | 4.14% | 0.01% | 40.58% | 55.21% | 0.06% |
| Assistant | 0.05% | 0.00% | 0.28% | 99.63% | 0.03% |
| Tool | 1.26% | 0.16% | 0.43% | 0.49% | 97.67% |

Probe coverage: 95/114 whole histories. 19 exceed native context and have no probe score; task completion still includes every attempt.

### SWE-bench Verified

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 60.65% | 5.56% | 18.06% | 0.00% | 15.74% |
| User | 1.21% | 91.31% | 3.84% | 1.34% | 2.30% |
| Reasoning | 0.02% | 0.88% | 98.09% | 0.82% | 0.18% |
| Assistant | 0.02% | 0.04% | 46.73% | 49.44% | 3.78% |
| Tool | 8.02% | 30.85% | 25.49% | 4.62% | 31.02% |

Probe coverage: 500/500 whole histories. 0 exceed native context and have no probe score; task completion still includes every attempt.
<!-- /fresh-rollout-confusions -->
