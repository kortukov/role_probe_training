# Confusion matrices

Rows are ground-truth roles; columns are predicted roles. Counts include message-content tokens only. Headers, special tokens and synthetic tool-schema context are excluded. `cot` denotes assistant reasoning; `assistant` denotes non-reasoning assistant content, including tool calls. A missing true role has no recall estimate.

## Supplementary metrics

**Averaging.** Mean conversation accuracy gives each conversation equal weight. Role-balanced accuracy averages recall equally over roles present in the dataset; recall is the fraction of tokens of a true role classified correctly. Uniform random prediction over the five possible classes has expected accuracy 20%.

**Segmentation.** Long histories are replayed in separate segments with context reset between them. “Segmented conversations” counts histories requiring more than one segment. “Unsegmented accuracy” is token accuracy restricted to histories that fit in one segment. It measures a different subset, not a controlled effect of segmentation. The segment size is documented in RUNNING.md.

| Dataset | Layer | Content tokens | Mean conversation accuracy | Role-balanced accuracy | Segmented conversations | Unsegmented accuracy |
|---|---:|---:|---:|---:|---:|---:|
| appworld_test_normal | 4 | 2,364,547 | 51.35% | 39.68% | 59 / 168 | 52.74% |
| appworld_test_normal | 8 | 2,364,547 | 21.63% | 43.45% | 59 / 168 | 23.57% |
| appworld_test_normal | 12 | 2,364,547 | 24.70% | 33.38% | 59 / 168 | 25.92% |
| appworld_test_normal | 16 | 2,364,547 | 31.81% | 29.11% | 59 / 168 | 33.15% |
| appworld_test_normal | 20 | 2,364,547 | 31.20% | 34.17% | 59 / 168 | 31.98% |

The [machine-readable results](metrics.json) also report mean true-role probability: the average probability assigned to the correct role, within each true role. This differs from accuracy, which uses the highest-probability class. Individual conversation results are included there and in the viewer.

## appworld_test_normal layer 4

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2702 | 10572 | 1876 | 11079 | 14763 |
| user | 43 | 6946 | 1349 | 3024 | 1755 |
| cot | 130 | 3918 | 88611 | 10901 | 83923 |
| assistant | 968 | 2408 | 148125 | 33339 | 724218 |
| tool | 4228 | 6214 | 88390 | 47817 | 1067248 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.6% | 25.8% | 4.6% | 27.0% | 36.0% |
| user | 0.3% | 53.0% | 10.3% | 23.1% | 13.4% |
| cot | 0.1% | 2.1% | 47.3% | 5.8% | 44.8% |
| assistant | 0.1% | 0.3% | 16.3% | 3.7% | 79.7% |
| tool | 0.3% | 0.5% | 7.3% | 3.9% | 87.9% |

## appworld_test_normal layer 8

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2575 | 6376 | 6843 | 0 | 25198 |
| user | 0 | 12269 | 226 | 2 | 620 |
| cot | 54 | 3762 | 174262 | 4697 | 4708 |
| assistant | 2548 | 4250 | 828457 | 47880 | 25923 |
| tool | 189514 | 285231 | 426501 | 79129 | 233522 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.3% | 15.6% | 16.7% | 0.0% | 61.5% |
| user | 0.0% | 93.5% | 1.7% | 0.0% | 4.7% |
| cot | 0.0% | 2.0% | 92.9% | 2.5% | 2.5% |
| assistant | 0.3% | 0.5% | 91.1% | 5.3% | 2.9% |
| tool | 15.6% | 23.5% | 35.1% | 6.5% | 19.2% |

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

## appworld_test_normal layer 16

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 773 | 12365 | 1657 | 0 | 26197 |
| user | 13 | 8656 | 81 | 1 | 4366 |
| cot | 1908 | 659 | 30344 | 13584 | 140988 |
| assistant | 38796 | 1981 | 183286 | 182065 | 502930 |
| tool | 205180 | 171014 | 38707 | 295427 | 503569 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.9% | 30.2% | 4.0% | 0.0% | 63.9% |
| user | 0.1% | 66.0% | 0.6% | 0.0% | 33.3% |
| cot | 1.0% | 0.4% | 16.2% | 7.2% | 75.2% |
| assistant | 4.3% | 0.2% | 20.2% | 20.0% | 55.3% |
| tool | 16.9% | 14.1% | 3.2% | 24.3% | 41.5% |

## appworld_test_normal layer 20

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 222 | 22269 | 4406 | 1 | 14094 |
| user | 6 | 11700 | 116 | 46 | 1249 |
| cot | 2345 | 14791 | 44191 | 11808 | 114348 |
| assistant | 23608 | 79256 | 212816 | 102913 | 490465 |
| tool | 138697 | 359602 | 45135 | 109306 | 561157 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.5% | 54.3% | 10.7% | 0.0% | 34.4% |
| user | 0.0% | 89.2% | 0.9% | 0.4% | 9.5% |
| cot | 1.3% | 7.9% | 23.6% | 6.3% | 61.0% |
| assistant | 2.6% | 8.7% | 23.4% | 11.3% | 54.0% |
| tool | 11.4% | 29.6% | 3.7% | 9.0% | 46.2% |
