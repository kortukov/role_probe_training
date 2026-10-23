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

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 3,666 | 814 | 1,586 | 1,124 | 15,810 |
| user | 1,165 | 26,664 | 11,732 | 18,147 | 26,408 |
| cot | 364 | 6,116 | 290,311 | 67,211 | 104,386 |
| assistant | 914 | 2,027 | 156,845 | 201,365 | 395,771 |
| tool | 9,808 | 13,456 | 258,184 | 178,367 | 984,731 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 15.94 | 3.54 | 6.90 | 4.89 | 68.74 |
| user | 1.38 | 31.70 | 13.95 | 21.57 | 31.39 |
| cot | 0.08 | 1.31 | 61.98 | 14.35 | 22.29 |
| assistant | 0.12 | 0.27 | 20.72 | 26.60 | 52.29 |
| tool | 0.68 | 0.93 | 17.87 | 12.35 | 68.17 |

### Layer 8

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 6,052 | 89 | 2,166 | 0 | 14,693 |
| user | 2,221 | 76,865 | 1,770 | 156 | 3,104 |
| cot | 79 | 8,279 | 372,046 | 78,735 | 9,249 |
| assistant | 1,452 | 2,210 | 112,944 | 636,005 | 4,311 |
| tool | 44,437 | 70,133 | 573,080 | 445,541 | 311,355 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 26.31 | 0.39 | 9.42 | 0.00 | 63.88 |
| user | 2.64 | 91.38 | 2.10 | 0.19 | 3.69 |
| cot | 0.02 | 1.77 | 79.43 | 16.81 | 1.97 |
| assistant | 0.19 | 0.29 | 14.92 | 84.03 | 0.57 |
| tool | 3.08 | 4.86 | 39.67 | 30.84 | 21.55 |

### Layer 12

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 1,506 | 1 | 13,089 | 1 | 8,403 |
| user | 184 | 67,164 | 10,334 | 199 | 6,235 |
| cot | 101 | 6,521 | 276,463 | 179,363 | 5,940 |
| assistant | 1,606 | 2,740 | 57,442 | 690,685 | 4,449 |
| tool | 39,484 | 58,116 | 560,082 | 632,381 | 154,483 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 6.55 | 0.00 | 56.91 | 0.00 | 36.53 |
| user | 0.22 | 79.85 | 12.29 | 0.24 | 7.41 |
| cot | 0.02 | 1.39 | 59.02 | 38.29 | 1.27 |
| assistant | 0.21 | 0.36 | 7.59 | 91.25 | 0.59 |
| tool | 2.73 | 4.02 | 38.77 | 43.78 | 10.69 |

### Layer 16

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 7,344 | 563 | 1,937 | 404 | 12,752 |
| user | 1,670 | 72,195 | 3,145 | 774 | 6,332 |
| cot | 579 | 546 | 237,648 | 156,513 | 73,102 |
| assistant | 1,876 | 1,509 | 10,210 | 724,373 | 18,954 |
| tool | 24,211 | 74,056 | 93,760 | 1,131,367 | 121,152 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 31.93 | 2.45 | 8.42 | 1.76 | 55.44 |
| user | 1.99 | 85.83 | 3.74 | 0.92 | 7.53 |
| cot | 0.12 | 0.12 | 50.74 | 33.42 | 15.61 |
| assistant | 0.25 | 0.20 | 1.35 | 95.70 | 2.50 |
| tool | 1.68 | 5.13 | 6.49 | 78.32 | 8.39 |

### Layer 20

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 2,348 | 1,202 | 4,858 | 30 | 14,562 |
| user | 356 | 80,588 | 371 | 294 | 2,507 |
| cot | 711 | 22,995 | 269,159 | 125,227 | 50,296 |
| assistant | 1,804 | 47,232 | 11,216 | 679,355 | 17,315 |
| tool | 34,599 | 495,013 | 80,744 | 619,097 | 215,093 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 10.21 | 5.23 | 21.12 | 0.13 | 63.31 |
| user | 0.42 | 95.81 | 0.44 | 0.35 | 2.98 |
| cot | 0.15 | 4.91 | 57.46 | 26.74 | 10.74 |
| assistant | 0.24 | 6.24 | 1.48 | 89.75 | 2.29 |
| tool | 2.40 | 34.27 | 5.59 | 42.86 | 14.89 |

Machine-readable results: [baseline_metrics.json](baseline_metrics.json). Execution metadata: [baseline_manifest.json](baseline_manifest.json).
