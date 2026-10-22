# Confusion matrices

Rows are ground-truth roles; columns are predicted roles. Counts include message-content tokens only. Headers, special tokens and synthetic tool-schema context are excluded. `cot` denotes assistant reasoning; `assistant` denotes non-reasoning assistant content, including tool calls. A missing true role has no recall estimate.

## Supplementary metrics

**Averaging.** Mean conversation accuracy gives each conversation equal weight. Role-balanced accuracy averages recall equally over roles present in the dataset; recall is the fraction of tokens of a true role classified correctly. Dolci has four observed roles; retail and SWE have five. Uniform random prediction over the five possible classes has expected accuracy 20%.

**Segmentation.** Long histories are replayed in separate segments with context reset between them. “Segmented conversations” counts histories requiring more than one segment. “Unsegmented accuracy” is token accuracy restricted to histories that fit in one segment. It measures a different subset, not a controlled effect of segmentation. The segment size is documented in RUNNING.md.

| Dataset | Layer | Content tokens | Mean conversation accuracy | Role-balanced accuracy | Segmented conversations | Unsegmented accuracy |
|---|---:|---:|---:|---:|---:|---:|
| dolci_tool_use | 12 | 1,516,257 | 43.84% | 40.37% | 11 / 1000 | 38.93% |
| tau2_retail | 12 | 2,876,950 | 31.60% | 33.08% | 79 / 114 | 35.66% |
| swe_bench | 12 | 7,265,153 | 28.43% | 34.79% | 144 / 500 | 24.59% |

The [machine-readable results](metrics.json) also report mean true-role probability: the average probability assigned to the correct role, within each true role. This differs from accuracy, which uses the highest-probability class. Individual conversation results are included there and in the viewer.

## dolci_tool_use layer 12

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1763 | 3483 | 34638 | 0 | 27846 |
| user | 128 | 47168 | 9445 | 19149 | 1694 |
| cot | 0 | 0 | 0 | 0 | 0 |
| assistant | 761 | 2247 | 37295 | 263127 | 3046 |
| tool | 443467 | 155398 | 206779 | 128533 | 130290 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.6% | 5.1% | 51.1% | 0.0% | 41.1% |
| user | 0.2% | 60.8% | 12.2% | 24.7% | 2.2% |
| cot | n/a | n/a | n/a | n/a | n/a |
| assistant | 0.2% | 0.7% | 12.2% | 85.9% | 1.0% |
| tool | 41.7% | 14.6% | 19.4% | 12.1% | 12.2% |

## tau2_retail layer 12

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 11566 | 59578 | 69960 | 0 | 28870 |
| user | 349 | 10025 | 4648 | 16937 | 13 |
| cot | 25163 | 5774 | 768623 | 1613144 | 1416 |
| assistant | 39 | 1 | 4054 | 68027 | 32 |
| tool | 2808 | 340 | 65941 | 117491 | 2151 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.8% | 35.1% | 41.2% | 0.0% | 17.0% |
| user | 1.1% | 31.4% | 14.5% | 53.0% | 0.0% |
| cot | 1.0% | 0.2% | 31.8% | 66.8% | 0.1% |
| assistant | 0.1% | 0.0% | 5.6% | 94.3% | 0.0% |
| tool | 1.5% | 0.2% | 34.9% | 62.3% | 1.1% |

## swe_bench layer 12

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1647 | 17242 | 10371 | 0 | 78740 |
| user | 1951 | 221967 | 5720 | 1402 | 3531 |
| cot | 105684 | 65607 | 1559267 | 1584601 | 53542 |
| assistant | 1905 | 1109 | 115582 | 52163 | 16857 |
| tool | 479975 | 1920490 | 602513 | 237871 | 125416 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.5% | 16.0% | 9.6% | 0.0% | 72.9% |
| user | 0.8% | 94.6% | 2.4% | 0.6% | 1.5% |
| cot | 3.1% | 1.9% | 46.3% | 47.0% | 1.6% |
| assistant | 1.0% | 0.6% | 61.6% | 27.8% | 9.0% |
| tool | 14.3% | 57.1% | 17.9% | 7.1% | 3.7% |
