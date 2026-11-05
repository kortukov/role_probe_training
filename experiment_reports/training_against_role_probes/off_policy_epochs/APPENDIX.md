# Retail checkpoint details

Confusion matrices count content tokens: rows are true roles and columns are predicted roles. The percentage matrix normalizes each true-role row. Reasoning is the `cot` label. Headers and synthetic schemas are excluded. Whole histories above native context are excluded only from role metrics.

## Epoch 0

Task success: 60/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch000/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 11172 | 64410 | 66918 | 0 | 27474 |
| User | 867 | 9093 | 2453 | 16539 | 2 |
| Reasoning | 27030 | 3924 | 254697 | 1164272 | 121 |
| Assistant | 5406 | 225 | 51025 | 377286 | 184 |
| Tool | 2342 | 640 | 60426 | 138475 | 3282 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 6.6 | 37.9 | 39.4 | 0.0 | 16.2 |
| User | 3.0 | 31.4 | 8.5 | 57.1 | 0.0 |
| Reasoning | 1.9 | 0.3 | 17.6 | 80.3 | 0.0 |
| Assistant | 1.2 | 0.1 | 11.8 | 86.9 | 0.0 |
| Tool | 1.1 | 0.3 | 29.5 | 67.5 | 1.6 |

Excluded histories: 0. The linked result records every exclusion and its token length.

## Epoch 1

Task success: 56/114. Role coverage: 111/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch001/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 13431 | 27972 | 88467 | 0 | 35631 |
| User | 513 | 11642 | 3215 | 15287 | 11 |
| Reasoning | 31197 | 4979 | 365458 | 1352821 | 495 |
| Assistant | 11496 | 497 | 61556 | 527785 | 276 |
| Tool | 1418 | 1265 | 67110 | 121821 | 7346 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 8.1 | 16.9 | 53.5 | 0.0 | 21.5 |
| User | 1.7 | 38.0 | 10.5 | 49.8 | 0.0 |
| Reasoning | 1.8 | 0.3 | 20.8 | 77.1 | 0.0 |
| Assistant | 1.9 | 0.1 | 10.2 | 87.7 | 0.0 |
| Tool | 0.7 | 0.6 | 33.7 | 61.2 | 3.7 |

Excluded histories: 3. The linked result records every exclusion and its token length.

## Epoch 2

Task success: 67/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch002/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 34884 | 31806 | 86526 | 0 | 16758 |
| User | 405 | 13752 | 2003 | 12710 | 15 |
| Reasoning | 20343 | 3885 | 362348 | 743963 | 1314 |
| Assistant | 4070 | 478 | 91716 | 274188 | 404 |
| Tool | 1021 | 2560 | 70379 | 85900 | 44766 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 20.5 | 18.7 | 50.9 | 0.0 | 9.9 |
| User | 1.4 | 47.6 | 6.9 | 44.0 | 0.1 |
| Reasoning | 1.8 | 0.3 | 32.0 | 65.7 | 0.1 |
| Assistant | 1.1 | 0.1 | 24.7 | 73.9 | 0.1 |
| Tool | 0.5 | 1.3 | 34.4 | 42.0 | 21.9 |

Excluded histories: 0. The linked result records every exclusion and its token length.
