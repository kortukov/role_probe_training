# Confusion matrices

Rows are ground-truth roles; columns are predicted roles. Counts include message-content tokens only. Headers, special tokens and synthetic tool-schema context are excluded. `cot` denotes assistant reasoning; `assistant` denotes non-reasoning assistant content, including tool calls. A missing true role has no recall estimate.

## Supplementary metrics

**Averaging.** Mean conversation accuracy gives each conversation equal weight. Role-balanced accuracy averages recall equally over roles present in the dataset; recall is the fraction of tokens of a true role classified correctly. Uniform random prediction over the five possible classes has expected accuracy 20%.

**Segmentation.** Long histories are replayed in separate segments with context reset between them. “Segmented conversations” counts histories requiring more than one segment. “Unsegmented accuracy” is token accuracy restricted to histories that fit in one segment. It measures a different subset, not a controlled effect of segmentation. The segment size is documented in RUNNING.md.

| Dataset | Layer | Content tokens | Mean conversation accuracy | Role-balanced accuracy | Segmented conversations | Unsegmented accuracy |
|---|---:|---:|---:|---:|---:|---:|
| tau2_retail | 4 | 2,876,950 | 21.88% | 26.33% | 79 / 114 | 28.78% |
| tau2_retail | 8 | 2,876,950 | 69.45% | 44.93% | 79 / 114 | 64.33% |
| tau2_retail | 12 | 2,876,950 | 31.60% | 33.08% | 79 / 114 | 35.66% |
| tau2_retail | 16 | 2,876,950 | 39.53% | 31.47% | 79 / 114 | 46.14% |
| tau2_retail | 20 | 2,876,950 | 60.83% | 40.91% | 79 / 114 | 47.01% |
| swe_bench | 4 | 7,265,153 | 55.97% | 34.52% | 144 / 500 | 63.08% |
| swe_bench | 8 | 7,265,153 | 36.76% | 40.37% | 144 / 500 | 27.24% |
| swe_bench | 12 | 7,265,153 | 28.43% | 34.79% | 144 / 500 | 24.59% |
| swe_bench | 16 | 7,265,153 | 23.00% | 34.25% | 144 / 500 | 22.09% |
| swe_bench | 20 | 7,265,153 | 34.95% | 39.35% | 144 / 500 | 30.48% |
| appworld_test_normal | 4 | 2,364,547 | 51.35% | 39.68% | 59 / 168 | 52.74% |
| appworld_test_normal | 8 | 2,364,547 | 21.63% | 43.45% | 59 / 168 | 23.57% |
| appworld_test_normal | 12 | 2,364,547 | 24.70% | 33.38% | 59 / 168 | 25.92% |
| appworld_test_normal | 16 | 2,364,547 | 31.81% | 29.11% | 59 / 168 | 33.15% |
| appworld_test_normal | 20 | 2,364,547 | 31.20% | 34.17% | 59 / 168 | 31.98% |

The [machine-readable results](metrics.json) also report mean true-role probability: the average probability assigned to the correct role, within each true role. This differs from accuracy, which uses the highest-probability class. Individual conversation results are included there and in the viewer.

## tau2_retail layer 4

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 600 | 19218 | 3274 | 15538 | 131344 |
| user | 56 | 3065 | 947 | 8367 | 19537 |
| cot | 22054 | 6532 | 312405 | 59011 | 2014118 |
| assistant | 30 | 22 | 3580 | 10773 | 57748 |
| tool | 83 | 294 | 6132 | 5115 | 177107 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.4% | 11.3% | 1.9% | 9.1% | 77.3% |
| user | 0.2% | 9.6% | 3.0% | 26.2% | 61.1% |
| cot | 0.9% | 0.3% | 12.9% | 2.4% | 83.4% |
| assistant | 0.0% | 0.0% | 5.0% | 14.9% | 80.0% |
| tool | 0.0% | 0.2% | 3.2% | 2.7% | 93.8% |

## tau2_retail layer 8

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 41379 | 19999 | 12072 | 0 | 96524 |
| user | 1497 | 8318 | 1132 | 20996 | 29 |
| cot | 52977 | 5723 | 1948097 | 390975 | 16348 |
| assistant | 19 | 2 | 7450 | 64656 | 26 |
| tool | 955 | 79 | 126386 | 53795 | 7516 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 24.3% | 11.8% | 7.1% | 0.0% | 56.8% |
| user | 4.7% | 26.0% | 3.5% | 65.7% | 0.1% |
| cot | 2.2% | 0.2% | 80.7% | 16.2% | 0.7% |
| assistant | 0.0% | 0.0% | 10.3% | 89.6% | 0.0% |
| tool | 0.5% | 0.0% | 67.0% | 28.5% | 4.0% |

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

## tau2_retail layer 16

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 15693 | 37610 | 20840 | 0 | 95831 |
| user | 1824 | 2344 | 1627 | 24319 | 1858 |
| cot | 150728 | 284 | 924083 | 1252489 | 86536 |
| assistant | 282 | 10 | 566 | 70513 | 782 |
| tool | 7496 | 517 | 9225 | 162459 | 9034 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 9.2% | 22.1% | 12.3% | 0.0% | 56.4% |
| user | 5.7% | 7.3% | 5.1% | 76.1% | 5.8% |
| cot | 6.2% | 0.0% | 38.3% | 51.9% | 3.6% |
| assistant | 0.4% | 0.0% | 0.8% | 97.7% | 1.1% |
| tool | 4.0% | 0.3% | 4.9% | 86.1% | 4.8% |

## tau2_retail layer 20

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2237 | 103682 | 19732 | 3 | 44320 |
| user | 2265 | 7108 | 735 | 20198 | 1666 |
| cot | 78492 | 28936 | 1852292 | 224999 | 229401 |
| assistant | 512 | 3436 | 639 | 66860 | 706 |
| tool | 26744 | 8905 | 24756 | 106453 | 21873 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.3% | 61.0% | 11.6% | 0.0% | 26.1% |
| user | 7.1% | 22.2% | 2.3% | 63.2% | 5.2% |
| cot | 3.3% | 1.2% | 76.7% | 9.3% | 9.5% |
| assistant | 0.7% | 4.8% | 0.9% | 92.7% | 1.0% |
| tool | 14.2% | 4.7% | 13.1% | 56.4% | 11.6% |

## swe_bench layer 4

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2956 | 18952 | 2500 | 12770 | 70822 |
| user | 5267 | 162997 | 5759 | 6947 | 53601 |
| cot | 13791 | 35057 | 411879 | 33184 | 2874790 |
| assistant | 568 | 4221 | 12170 | 8948 | 161709 |
| tool | 49836 | 464799 | 36704 | 8003 | 2806923 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.7% | 17.5% | 2.3% | 11.8% | 65.6% |
| user | 2.2% | 69.5% | 2.5% | 3.0% | 22.9% |
| cot | 0.4% | 1.0% | 12.2% | 1.0% | 85.3% |
| assistant | 0.3% | 2.2% | 6.5% | 4.8% | 86.2% |
| tool | 1.5% | 13.8% | 1.1% | 0.2% | 83.4% |

## swe_bench layer 8

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 8887 | 47002 | 18370 | 0 | 33741 |
| user | 1598 | 224175 | 5243 | 1345 | 2210 |
| cot | 95373 | 47258 | 2963310 | 191793 | 70967 |
| assistant | 2499 | 2797 | 156325 | 15624 | 10371 |
| tool | 440058 | 2548311 | 296868 | 21910 | 59118 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 8.2% | 43.5% | 17.0% | 0.0% | 31.2% |
| user | 0.7% | 95.6% | 2.2% | 0.6% | 0.9% |
| cot | 2.8% | 1.4% | 88.0% | 5.7% | 2.1% |
| assistant | 1.3% | 1.5% | 83.3% | 8.3% | 5.5% |
| tool | 13.1% | 75.7% | 8.8% | 0.7% | 1.8% |

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

## swe_bench layer 16

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2116 | 68549 | 0 | 4 | 37331 |
| user | 3653 | 217438 | 2516 | 3330 | 7634 |
| cot | 497042 | 34550 | 811861 | 1245848 | 779400 |
| assistant | 16912 | 16553 | 7525 | 72086 | 74540 |
| tool | 1295519 | 1351354 | 99421 | 146035 | 473936 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.0% | 63.5% | 0.0% | 0.0% | 34.6% |
| user | 1.6% | 92.7% | 1.1% | 1.4% | 3.3% |
| cot | 14.8% | 1.0% | 24.1% | 37.0% | 23.1% |
| assistant | 9.0% | 8.8% | 4.0% | 38.4% | 39.7% |
| tool | 38.5% | 40.1% | 3.0% | 4.3% | 14.1% |

## swe_bench layer 20

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 7 | 83276 | 3444 | 0 | 21273 |
| user | 3082 | 209961 | 10128 | 3521 | 7879 |
| cot | 243907 | 158157 | 2003239 | 437816 | 525582 |
| assistant | 6264 | 90937 | 9504 | 48108 | 32803 |
| tool | 799852 | 1226704 | 525102 | 69837 | 744770 |

Row-normalized percentages (the diagonal is role recall):

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.0% | 77.1% | 3.2% | 0.0% | 19.7% |
| user | 1.3% | 89.5% | 4.3% | 1.5% | 3.4% |
| cot | 7.2% | 4.7% | 59.5% | 13.0% | 15.6% |
| assistant | 3.3% | 48.5% | 5.1% | 25.6% | 17.5% |
| tool | 23.8% | 36.4% | 15.6% | 2.1% | 22.1% |

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
