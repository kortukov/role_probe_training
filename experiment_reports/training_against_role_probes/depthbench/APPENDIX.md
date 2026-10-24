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

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 222,803 | 69,812 | 4,259 | 13,392 | 338,734 |
| user | 3,679 | 102,083 | 35,301 | 38,974 | 27,983 |
| cot | 2,501 | 36,900 | 1,924,636 | 182,171 | 1,524,237 |
| assistant | 1,441 | 3,326 | 1,040,209 | 609,470 | 2,710,602 |
| tool | 1,316 | 4,413 | 1,455,689 | 275,491 | 2,190,719 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 34.33 | 10.76 | 0.66 | 2.06 | 52.19 |
| user | 1.77 | 49.07 | 16.97 | 18.74 | 13.45 |
| cot | 0.07 | 1.01 | 52.44 | 4.96 | 41.53 |
| assistant | 0.03 | 0.08 | 23.83 | 13.96 | 62.10 |
| tool | 0.03 | 0.11 | 37.06 | 7.01 | 55.78 |

### Layer 8

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 99,753 | 148,241 | 122,616 | 0 | 278,390 |
| user | 100 | 207,360 | 280 | 240 | 40 |
| cot | 305 | 63,237 | 3,513,413 | 54,945 | 38,545 |
| assistant | 3,667 | 8,763 | 1,812,600 | 2,508,542 | 31,476 |
| tool | 4,573 | 19,534 | 3,621,958 | 114,141 | 167,422 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 15.37 | 22.84 | 18.89 | 0.00 | 42.90 |
| user | 0.05 | 99.68 | 0.13 | 0.12 | 0.02 |
| cot | 0.01 | 1.72 | 95.72 | 1.50 | 1.05 |
| assistant | 0.08 | 0.20 | 41.53 | 57.47 | 0.72 |
| tool | 0.12 | 0.50 | 92.22 | 2.91 | 4.26 |

### Layer 12

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 1,672 | 39,614 | 246,785 | 0 | 360,929 |
| user | 20 | 191,340 | 14,600 | 120 | 1,940 |
| cot | 37 | 41,882 | 2,961,804 | 640,877 | 25,845 |
| assistant | 4,787 | 6,466 | 1,300,578 | 3,027,550 | 25,667 |
| tool | 6,019 | 60,856 | 3,324,675 | 360,138 | 175,940 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 0.26 | 6.10 | 38.03 | 0.00 | 55.61 |
| user | 0.01 | 91.98 | 7.02 | 0.06 | 0.93 |
| cot | 0.00 | 1.14 | 80.69 | 17.46 | 0.70 |
| assistant | 0.11 | 0.15 | 29.80 | 69.36 | 0.59 |
| tool | 0.15 | 1.55 | 84.65 | 9.17 | 4.48 |

### Layer 16

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 72,333 | 231,826 | 8,605 | 0 | 336,236 |
| user | 1,184 | 181,758 | 4,896 | 4,681 | 15,501 |
| cot | 16,460 | 17,378 | 2,589,880 | 176,386 | 870,341 |
| assistant | 24,389 | 11,266 | 294,720 | 3,764,393 | 270,280 |
| tool | 75,660 | 26,289 | 622,177 | 2,672,219 | 531,283 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 11.15 | 35.72 | 1.33 | 0.00 | 51.81 |
| user | 0.57 | 87.38 | 2.35 | 2.25 | 7.45 |
| cot | 0.45 | 0.47 | 70.56 | 4.81 | 23.71 |
| assistant | 0.56 | 0.26 | 6.75 | 86.24 | 6.19 |
| tool | 1.93 | 0.67 | 15.84 | 68.04 | 13.53 |

### Layer 20

| True / predicted | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 16,219 | 265,837 | 57,582 | 996 | 308,366 |
| user | 979 | 190,163 | 2,560 | 1,912 | 12,406 |
| cot | 20,909 | 317,302 | 2,007,568 | 194,243 | 1,130,423 |
| assistant | 40,548 | 490,477 | 325,832 | 3,298,845 | 209,346 |
| tool | 68,526 | 144,932 | 394,540 | 2,839,989 | 479,641 |

| True / predicted (%) | system | user | cot | assistant | tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| system | 2.50 | 40.96 | 8.87 | 0.15 | 47.51 |
| user | 0.47 | 91.42 | 1.23 | 0.92 | 5.96 |
| cot | 0.57 | 8.64 | 54.70 | 5.29 | 30.80 |
| assistant | 0.93 | 11.24 | 7.46 | 75.57 | 4.80 |
| tool | 1.74 | 3.69 | 10.05 | 72.31 | 12.21 |

Machine-readable results: [baseline_metrics.json](baseline_metrics.json). Execution metadata: [baseline_manifest.json](baseline_manifest.json).
