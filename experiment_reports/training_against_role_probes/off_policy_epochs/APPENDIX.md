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
