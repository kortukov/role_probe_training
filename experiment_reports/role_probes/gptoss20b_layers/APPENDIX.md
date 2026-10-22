# Supporting probe results

## Held-out confusion matrices

Rows are ground truth; columns are predicted roles. Counts cover content tokens in held-out documents. `cot` means assistant reasoning. Negative log likelihood (NLL) is the mean negative log probability assigned to the true role; lower is better.

### Layer 4

NLL: 1.2755.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 4944 | 2974 | 534 | 718 | 2132 |
| user | 2924 | 5946 | 400 | 517 | 1515 |
| cot | 1165 | 990 | 3891 | 1880 | 3376 |
| assistant | 1425 | 1084 | 1747 | 4126 | 2920 |
| tool | 1060 | 973 | 915 | 869 | 7485 |

### Layer 8

NLL: 0.5764.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 7765 | 1625 | 390 | 158 | 1364 |
| user | 1338 | 9451 | 73 | 25 | 415 |
| cot | 720 | 90 | 7798 | 859 | 1835 |
| assistant | 121 | 16 | 759 | 10176 | 230 |
| tool | 1327 | 228 | 967 | 180 | 8600 |

### Layer 20

NLL: 0.5792.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 7721 | 790 | 696 | 228 | 1867 |
| user | 511 | 10604 | 68 | 1 | 118 |
| cot | 450 | 60 | 7728 | 1161 | 1903 |
| assistant | 49 | 10 | 1042 | 10043 | 158 |
| tool | 1217 | 239 | 2000 | 365 | 7481 |
