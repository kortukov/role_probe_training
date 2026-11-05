# Retail checkpoint details

Confusion matrices count content tokens: rows are true roles and columns are predicted roles. The percentage matrix normalizes each true-role row. Reasoning is the `cot` label. Headers and synthetic schemas are excluded. Whole histories above native context are excluded only from role metrics.

## Epoch 0

Task success: 67/114. Role coverage: 113/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch000/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 11074 | 61698 | 67574 | 0 | 28137 |
| User | 705 | 8399 | 2397 | 17468 | 2 |
| Reasoning | 16553 | 3784 | 233457 | 956211 | 253 |
| Assistant | 3897 | 160 | 47005 | 385633 | 129 |
| Tool | 2105 | 446 | 58700 | 142149 | 2624 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 6.6 | 36.6 | 40.1 | 0.0 | 16.7 |
| User | 2.4 | 29.0 | 8.3 | 60.3 | 0.0 |
| Reasoning | 1.4 | 0.3 | 19.3 | 79.0 | 0.0 |
| Assistant | 0.9 | 0.0 | 10.8 | 88.3 | 0.0 |
| Tool | 1.0 | 0.2 | 28.5 | 69.0 | 1.3 |

Excluded histories: 1. The linked result records every exclusion and its token length.

## Epoch 1

Task success: 60/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch001/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 9234 | 31578 | 93822 | 0 | 35340 |
| User | 427 | 13760 | 3671 | 10691 | 6 |
| Reasoning | 33519 | 5089 | 412206 | 862167 | 364 |
| Assistant | 4956 | 561 | 85172 | 291623 | 251 |
| Tool | 1656 | 1579 | 98156 | 89145 | 9868 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 5.4 | 18.6 | 55.2 | 0.0 | 20.8 |
| User | 1.5 | 48.2 | 12.9 | 37.4 | 0.0 |
| Reasoning | 2.6 | 0.4 | 31.4 | 65.6 | 0.0 |
| Assistant | 1.3 | 0.1 | 22.3 | 76.2 | 0.1 |
| Tool | 0.8 | 0.8 | 49.0 | 44.5 | 4.9 |

Excluded histories: 0. The linked result records every exclusion and its token length.

## Epoch 2

Task success: 65/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch002/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 27132 | 33858 | 88008 | 0 | 20976 |
| User | 193 | 14162 | 1699 | 11145 | 14 |
| Reasoning | 26676 | 6118 | 606495 | 756334 | 1596 |
| Assistant | 971 | 360 | 58231 | 180374 | 456 |
| Tool | 1273 | 3149 | 75797 | 84818 | 36560 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 16.0 | 19.9 | 51.8 | 0.0 | 12.3 |
| User | 0.7 | 52.0 | 6.2 | 41.0 | 0.1 |
| Reasoning | 1.9 | 0.4 | 43.4 | 54.1 | 0.1 |
| Assistant | 0.4 | 0.1 | 24.2 | 75.0 | 0.2 |
| Tool | 0.6 | 1.6 | 37.6 | 42.1 | 18.1 |

Excluded histories: 0. The linked result records every exclusion and its token length.

## Epoch 3

Task success: 65/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch003/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 57114 | 32946 | 65892 | 0 | 14022 |
| User | 472 | 11828 | 3189 | 13220 | 10 |
| Reasoning | 31715 | 1934 | 826978 | 888519 | 452 |
| Assistant | 4284 | 75 | 86627 | 240335 | 100 |
| Tool | 807 | 626 | 79064 | 89769 | 29235 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 33.6 | 19.4 | 38.8 | 0.0 | 8.2 |
| User | 1.6 | 41.2 | 11.1 | 46.0 | 0.0 |
| Reasoning | 1.8 | 0.1 | 47.3 | 50.8 | 0.0 |
| Assistant | 1.3 | 0.0 | 26.1 | 72.5 | 0.0 |
| Tool | 0.4 | 0.3 | 39.6 | 45.0 | 14.7 |

Excluded histories: 0. The linked result records every exclusion and its token length.
