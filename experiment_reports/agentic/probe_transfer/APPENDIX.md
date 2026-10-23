# Confusion matrices

Rows are ground-truth roles; columns are predicted roles. Counts include message-content tokens only. Headers, special tokens and synthetic tool-schema context are excluded. `cot` denotes assistant reasoning; `assistant` denotes non-reasoning assistant content, including tool calls. A missing true role has no recall estimate.

## Supplementary metrics

**Averaging.** Mean conversation accuracy gives each conversation equal weight. Role-balanced accuracy averages recall equally over roles present in the dataset; recall is the fraction of tokens of a true role classified correctly. Uniform random prediction over the five possible classes has expected accuracy 20%.

**Segmentation.** Long histories are replayed in separate segments with context reset between them. “Segmented conversations” counts histories requiring more than one segment. “Unsegmented accuracy” is token accuracy restricted to histories that fit in one segment. It measures a different subset, not a controlled effect of segmentation. The segment size is documented in RUNNING.md.

| Dataset | Layer | Content tokens | Mean conversation accuracy | Role-balanced accuracy | Segmented conversations | Unsegmented accuracy |
|---|---:|---:|---:|---:|---:|---:|
| tau2_retail | 12 | 2,876,950 | 31.60% | 33.08% | 79 / 114 | 35.66% |
| swe_bench | 12 | 7,265,153 | 28.43% | 34.79% | 144 / 500 | 24.59% |
| appworld_test_normal | 12 | 2,364,547 | 24.70% | 33.38% | 59 / 168 | 25.92% |

The [machine-readable results](metrics.json) also report mean true-role probability: the average probability assigned to the correct role, within each true role. This differs from accuracy, which uses the highest-probability class. Individual conversation results are included there and in the viewer.

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

## appworld_test_normal layer 12

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 490 | 331 | 2826 | 0 | 37345 |
| user | 2 | 6513 | 246 | 0 | 6356 |
| cot | 21 | 6418 | 140009 | 27872 | 13163 |
| assistant | 1771 | 6038 | 630767 | 248911 | 21571 |
| tool | 124409 | 169943 | 530325 | 219153 | 170067 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.2% | 0.8% | 6.9% | 0.0% | 91.1% |
| user | 0.0% | 49.7% | 1.9% | 0.0% | 48.5% |
| cot | 0.0% | 3.4% | 74.7% | 14.9% | 7.0% |
| assistant | 0.2% | 0.7% | 69.4% | 27.4% | 2.4% |
| tool | 10.2% | 14.0% | 43.7% | 18.1% | 14.0% |
