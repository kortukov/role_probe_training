# Role metrics and coverage

Rows in each confusion matrix are true roles; columns are predicted roles. Values are row-normalized percentages. Zero-support rows are marked unavailable. All metrics use whole histories.

## Base

Coverage: 500/500 histories; 0 above native context.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.9 | 22.2 | 17.1 | 0.0 | 59.7 |
| user | 0.5 | 96.6 | 1.2 | 0.2 | 1.5 |
| cot | 0.3 | 3.7 | 72.0 | 20.3 | 3.8 |
| assistant | 0.3 | 1.1 | 66.4 | 27.2 | 4.9 |
| tool | 22.2 | 41.8 | 18.3 | 13.2 | 4.4 |

## AppWorld-trained

Pending.

## ASTRA-trained

Pending.

## DepthBench-trained

Pending.

## AppWorld + agentic KL (800)

Pending.

## AppWorld + same-batch KL

Pending.
