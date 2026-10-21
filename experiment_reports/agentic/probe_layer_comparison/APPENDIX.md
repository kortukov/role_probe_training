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

## dolci_tool_use layer 16

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 5915 | 31445 | 1122 | 122 | 29126 |
| user | 1030 | 35378 | 3163 | 34355 | 3658 |
| cot | 0 | 0 | 0 | 0 | 0 |
| assistant | 2173 | 754 | 15433 | 281559 | 6557 |
| tool | 479576 | 151439 | 111017 | 187773 | 134662 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 8.7% | 46.4% | 1.7% | 0.2% | 43.0% |
| user | 1.3% | 45.6% | 4.1% | 44.3% | 4.7% |
| cot | n/a | n/a | n/a | n/a | n/a |
| assistant | 0.7% | 0.2% | 5.0% | 91.9% | 2.1% |
| tool | 45.1% | 14.2% | 10.4% | 17.6% | 12.7% |

## dolci_tool_use layer 4

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 13293 | 16360 | 4669 | 2588 | 30820 |
| user | 797 | 30365 | 2587 | 32496 | 11339 |
| cot | 0 | 0 | 0 | 0 | 0 |
| assistant | 1230 | 3270 | 49949 | 106475 | 145552 |
| tool | 10925 | 8670 | 78401 | 117325 | 849146 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 19.6% | 24.2% | 6.9% | 3.8% | 45.5% |
| user | 1.0% | 39.1% | 3.3% | 41.9% | 14.6% |
| cot | n/a | n/a | n/a | n/a | n/a |
| assistant | 0.4% | 1.1% | 16.3% | 34.7% | 47.5% |
| tool | 1.0% | 0.8% | 7.4% | 11.0% | 79.8% |

## dolci_tool_use layer 8

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 15575 | 8816 | 17275 | 0 | 26064 |
| user | 382 | 61438 | 505 | 15127 | 132 |
| cot | 0 | 0 | 0 | 0 | 0 |
| assistant | 964 | 4015 | 44896 | 255644 | 957 |
| tool | 423159 | 257558 | 165566 | 110651 | 107533 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 23.0% | 13.0% | 25.5% | 0.0% | 38.5% |
| user | 0.5% | 79.2% | 0.7% | 19.5% | 0.2% |
| cot | n/a | n/a | n/a | n/a | n/a |
| assistant | 0.3% | 1.3% | 14.6% | 83.4% | 0.3% |
| tool | 39.8% | 24.2% | 15.6% | 10.4% | 10.1% |

## dolci_tool_use layer 20

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1720 | 47330 | 2592 | 35 | 16053 |
| user | 755 | 46767 | 1475 | 24380 | 4207 |
| cot | 0 | 0 | 0 | 0 | 0 |
| assistant | 1190 | 18840 | 16456 | 259179 | 10811 |
| tool | 342099 | 315085 | 79642 | 135208 | 192433 |

Row-normalized percentages:

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.5% | 69.9% | 3.8% | 0.1% | 23.7% |
| user | 1.0% | 60.3% | 1.9% | 31.4% | 5.4% |
| cot | n/a | n/a | n/a | n/a | n/a |
| assistant | 0.4% | 6.1% | 5.4% | 84.6% | 3.5% |
| tool | 32.1% | 29.6% | 7.5% | 12.7% | 18.1% |
