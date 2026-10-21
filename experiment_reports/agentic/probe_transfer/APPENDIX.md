# Confusion matrices

Rows are ground truth; columns are predictions. Counts exclude headers, special tokens and synthetic tool-schema context. Missing roles have no recall estimate.

## dolci_tool_use layer 12

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1763 | 3483 | 34638 | 0 | 27846 |
| user | 128 | 47168 | 9445 | 19149 | 1694 |
| cot | 0 | 0 | 0 | 0 | 0 |
| assistant | 761 | 2247 | 37295 | 263127 | 3046 |
| tool | 443467 | 155398 | 206779 | 128533 | 130290 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.6% | 5.1% | 51.1% | 0.0% | 41.1% |
| user | 0.2% | 60.8% | 12.2% | 24.7% | 2.2% |
| cot | n/a | n/a | n/a | n/a | n/a |
| assistant | 0.2% | 0.7% | 12.2% | 85.9% | 1.0% |
| tool | 41.7% | 14.6% | 19.4% | 12.1% | 12.2% |
