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

## Epoch 1

Task success: 56/114. Role coverage: 111/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch001/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 13431 | 27972 | 88467 | 0 | 35631 |
| User | 513 | 11642 | 3215 | 15287 | 11 |
| Reasoning | 31197 | 4979 | 365458 | 1352821 | 495 |
| Assistant | 11496 | 497 | 61556 | 527785 | 276 |
| Tool | 1418 | 1265 | 67110 | 121821 | 7346 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 8.1 | 16.9 | 53.5 | 0.0 | 21.5 |
| User | 1.7 | 38.0 | 10.5 | 49.8 | 0.0 |
| Reasoning | 1.8 | 0.3 | 20.8 | 77.1 | 0.0 |
| Assistant | 1.9 | 0.1 | 10.2 | 87.7 | 0.0 |
| Tool | 0.7 | 0.6 | 33.7 | 61.2 | 3.7 |

Excluded histories: 3. The linked result records every exclusion and its token length.

## Epoch 2

Task success: 67/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch002/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 34884 | 31806 | 86526 | 0 | 16758 |
| User | 405 | 13752 | 2003 | 12710 | 15 |
| Reasoning | 20343 | 3885 | 362348 | 743963 | 1314 |
| Assistant | 4070 | 478 | 91716 | 274188 | 404 |
| Tool | 1021 | 2560 | 70379 | 85900 | 44766 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 20.5 | 18.7 | 50.9 | 0.0 | 9.9 |
| User | 1.4 | 47.6 | 6.9 | 44.0 | 0.1 |
| Reasoning | 1.8 | 0.3 | 32.0 | 65.7 | 0.1 |
| Assistant | 1.1 | 0.1 | 24.7 | 73.9 | 0.1 |
| Tool | 0.5 | 1.3 | 34.4 | 42.0 | 21.9 |

Excluded histories: 0. The linked result records every exclusion and its token length.

## Epoch 3

Task success: 55/114. Role coverage: 113/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch003/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 56500 | 34917 | 67009 | 113 | 9944 |
| User | 374 | 13004 | 2370 | 12338 | 11 |
| Reasoning | 14281 | 2569 | 519274 | 506094 | 664 |
| Assistant | 6118 | 225 | 90654 | 234537 | 213 |
| Tool | 988 | 1094 | 65332 | 77366 | 51042 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 33.5 | 20.7 | 39.8 | 0.1 | 5.9 |
| User | 1.3 | 46.3 | 8.4 | 43.9 | 0.0 |
| Reasoning | 1.4 | 0.2 | 49.8 | 48.5 | 0.1 |
| Assistant | 1.8 | 0.1 | 27.3 | 70.7 | 0.1 |
| Tool | 0.5 | 0.6 | 33.4 | 39.5 | 26.1 |

Excluded histories: 1. The linked result records every exclusion and its token length.

## Epoch 4

Task success: 67/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch004/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 69996 | 26676 | 64980 | 0 | 8322 |
| User | 967 | 10330 | 3921 | 14655 | 6 |
| Reasoning | 41783 | 925 | 652150 | 781828 | 431 |
| Assistant | 8400 | 47 | 155330 | 341335 | 76 |
| Tool | 1379 | 357 | 66491 | 115632 | 28118 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 41.2 | 15.7 | 38.2 | 0.0 | 4.9 |
| User | 3.2 | 34.6 | 13.1 | 49.0 | 0.0 |
| Reasoning | 2.8 | 0.1 | 44.2 | 52.9 | 0.0 |
| Assistant | 1.7 | 0.0 | 30.7 | 67.6 | 0.0 |
| Tool | 0.7 | 0.2 | 31.4 | 54.5 | 13.3 |

Excluded histories: 0. The linked result records every exclusion and its token length.

## Epoch 5

Task success: 70/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch005/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 69882 | 21660 | 69768 | 0 | 8664 |
| User | 773 | 9308 | 7320 | 13284 | 7 |
| Reasoning | 18045 | 403 | 948303 | 573213 | 135 |
| Assistant | 3377 | 6 | 171965 | 221402 | 159 |
| Tool | 1325 | 278 | 104002 | 87998 | 17212 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 41.1 | 12.7 | 41.0 | 0.0 | 5.1 |
| User | 2.5 | 30.3 | 23.8 | 43.3 | 0.0 |
| Reasoning | 1.2 | 0.0 | 61.6 | 37.2 | 0.0 |
| Assistant | 0.9 | 0.0 | 43.3 | 55.8 | 0.0 |
| Tool | 0.6 | 0.1 | 49.3 | 41.7 | 8.2 |

Excluded histories: 0. The linked result records every exclusion and its token length.

## Epoch 6

Task success: 63/114. Role coverage: 113/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch006/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 70399 | 17176 | 72094 | 339 | 8475 |
| User | 1725 | 8049 | 8775 | 12660 | 7 |
| Reasoning | 38604 | 150 | 1090212 | 670697 | 172 |
| Assistant | 8507 | 5 | 215661 | 348973 | 8 |
| Tool | 1184 | 125 | 89350 | 100294 | 10202 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 41.8 | 10.2 | 42.8 | 0.2 | 5.0 |
| User | 5.5 | 25.8 | 28.1 | 40.6 | 0.0 |
| Reasoning | 2.1 | 0.0 | 60.6 | 37.3 | 0.0 |
| Assistant | 1.5 | 0.0 | 37.6 | 60.9 | 0.0 |
| Tool | 0.6 | 0.1 | 44.4 | 49.9 | 5.1 |

Excluded histories: 1. The linked result records every exclusion and its token length.

## Epoch 7

Task success: 61/114. Role coverage: 113/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch007/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 69495 | 18306 | 72433 | 0 | 8249 |
| User | 1508 | 8367 | 7813 | 13187 | 8 |
| Reasoning | 42965 | 213 | 1196354 | 791873 | 310 |
| Assistant | 5074 | 6 | 174108 | 274880 | 9 |
| Tool | 1605 | 147 | 110830 | 80451 | 5824 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 41.2 | 10.9 | 43.0 | 0.0 | 4.9 |
| User | 4.9 | 27.1 | 25.3 | 42.7 | 0.0 |
| Reasoning | 2.1 | 0.0 | 58.9 | 39.0 | 0.0 |
| Assistant | 1.1 | 0.0 | 38.3 | 60.5 | 0.0 |
| Tool | 0.8 | 0.1 | 55.7 | 40.5 | 2.9 |

Excluded histories: 1. The linked result records every exclusion and its token length.

## Epoch 8

Task success: 63/114. Role coverage: 109/114 histories. Native context: 131,072 tokens. [Counts and provenance](checkpoints/epoch008/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 67471 | 15478 | 69978 | 0 | 9592 |
| User | 1367 | 6717 | 7917 | 12675 | 8 |
| Reasoning | 30016 | 69 | 960459 | 774031 | 52 |
| Assistant | 1989 | 0 | 172374 | 299185 | 12 |
| Tool | 1088 | 113 | 98408 | 83637 | 3217 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 41.5 | 9.5 | 43.1 | 0.0 | 5.9 |
| User | 4.8 | 23.4 | 27.6 | 44.2 | 0.0 |
| Reasoning | 1.7 | 0.0 | 54.4 | 43.9 | 0.0 |
| Assistant | 0.4 | 0.0 | 36.4 | 63.2 | 0.0 |
| Tool | 0.6 | 0.1 | 52.8 | 44.9 | 1.7 |

Excluded histories: 5. The linked result records every exclusion and its token length.
