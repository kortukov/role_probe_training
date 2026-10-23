# Supplementary results

Rows denote true metadata-derived roles;
columns denote predicted roles. Counts and row-normalized percentages are reported
separately.

## Training-corpus baseline confusions

Rows are ground-truth content roles; columns are predicted roles. Counts include
all metadata-labeled content tokens from the 2,700 full trajectories.

### Layer 4

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 45900 | 167400 | 29700 | 180900 | 234900 |
| user | 480 | 95370 | 18150 | 48180 | 25830 |
| cot | 1560 | 71385 | 1268893 | 145010 | 1070890 |
| assistant | 9571 | 31141 | 1327140 | 296596 | 11365375 |
| tool | 23951 | 23882 | 1308743 | 743815 | 16286191 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 7.0% | 25.4% | 4.5% | 27.5% | 35.7% |
| user | 0.3% | 50.7% | 9.7% | 25.6% | 13.7% |
| cot | 0.1% | 2.8% | 49.6% | 5.7% | 41.9% |
| assistant | 0.1% | 0.2% | 10.2% | 2.3% | 87.2% |
| tool | 0.1% | 0.1% | 7.1% | 4.0% | 88.6% |

### Layer 8

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 43200 | 102600 | 110700 | 0 | 402300 |
| user | 0 | 174360 | 3270 | 0 | 10380 |
| cot | 1592 | 64965 | 2344683 | 57747 | 88751 |
| assistant | 38474 | 64966 | 11300701 | 1109900 | 515782 |
| tool | 2715273 | 4633141 | 5982347 | 1290331 | 3765490 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.6% | 15.6% | 16.8% | 0.0% | 61.1% |
| user | 0.0% | 92.7% | 1.7% | 0.0% | 5.5% |
| cot | 0.1% | 2.5% | 91.7% | 2.3% | 3.5% |
| assistant | 0.3% | 0.5% | 86.7% | 8.5% | 4.0% |
| tool | 14.8% | 25.2% | 32.5% | 7.0% | 20.5% |

### Layer 12

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 8100 | 5400 | 48600 | 0 | 596700 |
| user | 0 | 83520 | 2790 | 0 | 101700 |
| cot | 555 | 104237 | 1907122 | 333831 | 211993 |
| assistant | 15377 | 97072 | 8225590 | 4390467 | 301317 |
| tool | 1457323 | 3044718 | 7620092 | 3558629 | 2705820 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.2% | 0.8% | 7.4% | 0.0% | 90.6% |
| user | 0.0% | 44.4% | 1.5% | 0.0% | 54.1% |
| cot | 0.0% | 4.1% | 74.6% | 13.1% | 8.3% |
| assistant | 0.1% | 0.7% | 63.1% | 33.7% | 2.3% |
| tool | 7.9% | 16.6% | 41.4% | 19.4% | 14.7% |

### Layer 16

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 16200 | 189000 | 27000 | 0 | 426600 |
| user | 300 | 114630 | 1770 | 30 | 71280 |
| cot | 15922 | 11265 | 402844 | 156409 | 1971298 |
| assistant | 317051 | 17674 | 2421353 | 2941681 | 7332064 |
| tool | 2273437 | 2738595 | 553341 | 4504912 | 8316297 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.5% | 28.7% | 4.1% | 0.0% | 64.8% |
| user | 0.2% | 61.0% | 0.9% | 0.0% | 37.9% |
| cot | 0.6% | 0.4% | 15.8% | 6.1% | 77.1% |
| assistant | 2.4% | 0.1% | 18.6% | 22.6% | 56.3% |
| tool | 12.4% | 14.9% | 3.0% | 24.5% | 45.2% |

### Layer 20

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 5400 | 361800 | 70200 | 0 | 221400 |
| user | 150 | 166230 | 2160 | 330 | 19140 |
| cot | 21718 | 186307 | 597219 | 131100 | 1621394 |
| assistant | 250695 | 1138448 | 2845886 | 1480014 | 7314780 |
| tool | 1660106 | 5730382 | 581705 | 1546774 | 8867615 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.8% | 54.9% | 10.7% | 0.0% | 33.6% |
| user | 0.1% | 88.4% | 1.1% | 0.2% | 10.2% |
| cot | 0.8% | 7.3% | 23.3% | 5.1% | 63.4% |
| assistant | 1.9% | 8.7% | 21.8% | 11.4% | 56.1% |
| tool | 9.0% | 31.2% | 3.2% | 8.4% | 48.2% |

<!-- development-results -->
## Development confusions

Rows are true content roles; columns are predicted roles. Values are token counts.

### Before training

| True / predicted | system | user | reasoning | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 24 | 16 | 144 | 0 | 1768 |
| user | 0 | 218 | 9 | 0 | 304 |
| reasoning | 1 | 397 | 4601 | 403 | 542 |
| assistant | 21 | 253 | 21359 | 15331 | 833 |
| tool | 3447 | 9508 | 22515 | 9057 | 8961 |

Row-normalized percentages:

| True / predicted | system | user | reasoning | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.2% | 0.8% | 7.4% | 0.0% | 90.6% |
| user | 0.0% | 41.1% | 1.7% | 0.0% | 57.3% |
| reasoning | 0.0% | 6.7% | 77.4% | 6.8% | 9.1% |
| assistant | 0.1% | 0.7% | 56.5% | 40.6% | 2.2% |
| tool | 6.4% | 17.8% | 42.1% | 16.9% | 16.8% |

### After update 10

| True / predicted | system | user | reasoning | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1704 | 0 | 0 | 0 | 248 |
| user | 95 | 147 | 2 | 1 | 286 |
| reasoning | 1 | 50 | 784 | 4315 | 794 |
| assistant | 0 | 16 | 615 | 33947 | 3219 |
| tool | 5 | 15 | 53 | 475 | 52940 |

Row-normalized percentages:

| True / predicted | system | user | reasoning | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 87.3% | 0.0% | 0.0% | 0.0% | 12.7% |
| user | 17.9% | 27.7% | 0.4% | 0.2% | 53.9% |
| reasoning | 0.0% | 0.8% | 13.2% | 72.6% | 13.4% |
| assistant | 0.0% | 0.0% | 1.6% | 89.8% | 8.5% |
| tool | 0.0% | 0.0% | 0.1% | 0.9% | 99.0% |
<!-- /development-results -->
