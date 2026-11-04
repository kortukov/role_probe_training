# Role metrics and coverage

Rows in each confusion matrix are true roles; columns are predicted roles. Values are row-normalized percentages. Zero-support rows are marked unavailable. All metrics use whole histories.

## Base

Coverage: 481/500 histories; 19 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.9 | 22.2 | 17.1 | 0.0 | 59.7 |
| user | 0.5 | 83.1 | 1.4 | 13.7 | 1.3 |
| cot | 0.3 | 1.5 | 44.9 | 51.8 | 1.5 |
| assistant | 0.2 | 0.1 | 14.6 | 84.6 | 0.5 |
| tool | 18.9 | 17.3 | 13.1 | 48.8 | 1.9 |

## AppWorld-trained

Coverage: 500/500 histories; 0 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 39.8 | 50.0 | 9.7 | 0.0 | 0.5 |
| user | 0.2 | 93.8 | 1.5 | 1.5 | 3.0 |
| cot | 0.0 | 1.7 | 98.0 | 0.1 | 0.2 |
| assistant | 0.0 | 0.0 | 1.6 | 90.6 | 7.8 |
| tool | 2.2 | 10.1 | 2.2 | 0.0 | 85.5 |

## ASTRA-trained

Coverage: 498/500 histories; 2 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 19.0 | 16.2 | 3.2 | 0.0 | 61.6 |
| user | 0.7 | 84.8 | 3.6 | 5.4 | 5.6 |
| cot | 0.1 | 0.4 | 94.7 | 4.8 | 0.0 |
| assistant | 0.0 | 0.0 | 62.3 | 36.9 | 0.7 |
| tool | 8.1 | 1.0 | 20.0 | 4.6 | 66.2 |

## DepthBench-trained

Coverage: 498/500 histories; 2 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 60.6 | 5.6 | 18.1 | 0.0 | 15.7 |
| user | 1.2 | 89.2 | 4.4 | 2.7 | 2.6 |
| cot | 0.0 | 0.7 | 97.3 | 1.8 | 0.2 |
| assistant | 0.0 | 0.0 | 47.8 | 49.6 | 2.6 |
| tool | 7.1 | 28.3 | 30.1 | 4.8 | 29.7 |

## AppWorld + agentic KL (800)

Coverage: 493/500 histories; 7 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 36.6 | 54.2 | 6.0 | 0.5 | 2.8 |
| user | 0.6 | 79.0 | 1.3 | 14.1 | 5.1 |
| cot | 0.0 | 0.7 | 96.5 | 1.6 | 1.2 |
| assistant | 0.0 | 0.0 | 2.9 | 93.9 | 3.2 |
| tool | 0.1 | 1.9 | 1.8 | 1.5 | 94.8 |

## AppWorld + same-batch KL

Coverage: 485/500 histories; 15 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 38.0 | 53.2 | 5.1 | 0.5 | 3.2 |
| user | 0.6 | 66.6 | 1.8 | 26.7 | 4.3 |
| cot | 0.0 | 1.1 | 93.0 | 2.6 | 3.3 |
| assistant | 0.0 | 0.0 | 2.0 | 93.7 | 4.3 |
| tool | 0.1 | 2.1 | 0.8 | 1.5 | 95.4 |

## AppWorld + same-batch KL (27,000)

Coverage: 485/500 histories; 15 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 82.4 | 15.3 | 0.9 | 0.0 | 1.4 |
| user | 11.7 | 58.9 | 0.5 | 22.2 | 6.7 |
| cot | 0.2 | 0.9 | 98.0 | 0.9 | 0.1 |
| assistant | 1.4 | 0.8 | 1.1 | 95.8 | 1.0 |
| tool | 0.1 | 0.0 | 0.0 | 0.0 | 99.8 |
