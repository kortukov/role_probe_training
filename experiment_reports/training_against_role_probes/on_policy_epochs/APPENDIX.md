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
