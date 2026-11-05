# Appendix: confusion matrices and diagnostics

Rows are true roles, columns predicted roles, as percentages of each true role's
held-out tokens. Both collections are role-balanced by construction.


## C4/Dolma probe, C4/Dolma held-out


**Layer 4** (accuracy 46.7%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 43.9 | 26.4 | 4.7 | 6.3 | 18.7 |
| user | 25.9 | 52.7 | 3.5 | 4.5 | 13.4 |
| cot | 10.4 | 8.7 | 34.4 | 16.7 | 29.8 |
| assistant | 12.7 | 9.6 | 15.3 | 36.5 | 25.9 |
| tool | 9.4 | 8.6 | 8.1 | 7.7 | 66.1 |

**Layer 8** (accuracy 77.5%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 69.0 | 14.3 | 3.4 | 1.4 | 11.9 |
| user | 12.0 | 83.6 | 0.6 | 0.2 | 3.6 |
| cot | 6.4 | 0.8 | 68.9 | 7.6 | 16.3 |
| assistant | 1.1 | 0.2 | 6.8 | 89.9 | 2.1 |
| tool | 11.7 | 1.9 | 8.5 | 1.6 | 76.2 |

**Layer 12** (accuracy 68.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 63.9 | 14.2 | 5.9 | 3.9 | 12.1 |
| user | 16.7 | 74.7 | 2.0 | 0.7 | 5.9 |
| cot | 11.7 | 3.2 | 55.8 | 13.0 | 16.4 |
| assistant | 2.6 | 0.2 | 7.6 | 88.1 | 1.4 |
| tool | 18.1 | 5.7 | 10.8 | 3.4 | 62.0 |

**Layer 16** (accuracy 84.2%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 81.9 | 5.6 | 1.6 | 0.9 | 9.9 |
| user | 4.9 | 94.1 | 0.1 | 0.0 | 0.9 |
| cot | 3.1 | 0.4 | 75.7 | 6.8 | 13.9 |
| assistant | 0.2 | 0.2 | 5.1 | 94.1 | 0.5 |
| tool | 8.5 | 2.2 | 12.3 | 1.9 | 75.2 |

**Layer 20** (accuracy 77.0%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 68.1 | 6.8 | 6.1 | 2.1 | 16.8 |
| user | 4.4 | 94.0 | 0.7 | 0.0 | 0.9 |
| cot | 3.9 | 0.5 | 68.3 | 10.5 | 16.7 |
| assistant | 0.4 | 0.1 | 9.4 | 88.7 | 1.5 |
| tool | 10.8 | 2.2 | 17.8 | 3.3 | 66.0 |

## C4/Dolma probe, Toucan held-out


**Layer 4** (accuracy 40.6%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 22.3 | 12.9 | 13.1 | 27.6 | 24.2 |
| user | 10.7 | 32.3 | 11.7 | 25.1 | 20.1 |
| cot | 1.8 | 3.2 | 42.5 | 32.5 | 20.1 |
| assistant | 2.4 | 3.6 | 21.1 | 52.0 | 20.9 |
| tool | 4.1 | 3.5 | 12.4 | 25.8 | 54.2 |

**Layer 8** (accuracy 61.8%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 30.3 | 27.1 | 12.0 | 3.8 | 26.7 |
| user | 9.8 | 73.8 | 3.5 | 1.4 | 11.5 |
| cot | 2.4 | 4.4 | 66.9 | 12.7 | 13.6 |
| assistant | 2.3 | 4.0 | 15.9 | 70.7 | 7.1 |
| tool | 10.1 | 8.6 | 10.5 | 3.4 | 67.4 |

**Layer 12** (accuracy 55.2%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 41.4 | 22.1 | 13.7 | 3.9 | 18.9 |
| user | 11.7 | 71.7 | 4.1 | 0.9 | 11.6 |
| cot | 10.7 | 7.1 | 47.9 | 21.4 | 13.0 |
| assistant | 6.7 | 3.4 | 10.6 | 72.4 | 6.9 |
| tool | 27.5 | 11.8 | 15.3 | 2.7 | 42.8 |

**Layer 16** (accuracy 63.0%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 50.0 | 18.6 | 8.7 | 4.0 | 18.7 |
| user | 3.8 | 89.8 | 1.7 | 0.2 | 4.4 |
| cot | 7.0 | 4.3 | 47.5 | 26.0 | 15.1 |
| assistant | 4.4 | 1.3 | 7.3 | 79.8 | 7.1 |
| tool | 29.4 | 8.3 | 11.7 | 2.5 | 48.1 |

**Layer 20** (accuracy 52.2%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 24.7 | 44.7 | 14.0 | 6.9 | 9.7 |
| user | 1.5 | 95.2 | 1.6 | 0.4 | 1.4 |
| cot | 5.8 | 17.2 | 39.7 | 24.3 | 12.9 |
| assistant | 3.8 | 6.9 | 8.4 | 75.6 | 5.3 |
| tool | 13.0 | 36.7 | 18.3 | 6.4 | 25.6 |

## Toucan probe, C4/Dolma held-out


**Layer 4** (accuracy 35.0%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 44.9 | 45.4 | 1.8 | 2.5 | 5.4 |
| user | 25.9 | 67.3 | 1.3 | 1.9 | 3.6 |
| cot | 27.8 | 30.9 | 18.1 | 13.0 | 10.2 |
| assistant | 29.1 | 32.4 | 6.5 | 22.8 | 9.2 |
| tool | 29.0 | 30.8 | 7.1 | 11.5 | 21.6 |

**Layer 8** (accuracy 58.4%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 35.2 | 28.7 | 7.4 | 15.9 | 12.8 |
| user | 6.1 | 75.6 | 3.7 | 8.2 | 6.4 |
| cot | 7.6 | 2.7 | 58.6 | 22.7 | 8.4 |
| assistant | 1.1 | 0.4 | 6.6 | 90.5 | 1.5 |
| tool | 16.4 | 12.1 | 20.9 | 18.6 | 32.1 |

**Layer 12** (accuracy 52.5%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 38.1 | 15.2 | 10.9 | 20.6 | 15.3 |
| user | 9.7 | 62.5 | 6.8 | 10.7 | 10.3 |
| cot | 7.5 | 5.9 | 51.6 | 25.9 | 9.0 |
| assistant | 1.0 | 0.5 | 12.5 | 84.6 | 1.4 |
| tool | 16.3 | 13.1 | 22.6 | 22.4 | 25.5 |

**Layer 16** (accuracy 65.3%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 69.6 | 12.0 | 6.4 | 10.2 | 1.9 |
| user | 6.5 | 89.9 | 1.9 | 1.1 | 0.6 |
| cot | 9.6 | 2.1 | 68.3 | 18.5 | 1.6 |
| assistant | 0.5 | 0.1 | 10.1 | 89.3 | 0.0 |
| tool | 31.5 | 5.5 | 33.0 | 20.7 | 9.3 |

**Layer 20** (accuracy 58.6%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 56.8 | 9.0 | 13.5 | 18.7 | 1.9 |
| user | 8.8 | 85.3 | 3.1 | 2.0 | 0.7 |
| cot | 8.2 | 1.7 | 54.5 | 33.4 | 2.2 |
| assistant | 0.9 | 0.0 | 10.0 | 89.1 | 0.0 |
| tool | 25.1 | 3.9 | 32.2 | 31.7 | 7.1 |

## Toucan probe, Toucan held-out


**Layer 4** (accuracy 54.1%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 43.2 | 26.5 | 9.5 | 9.3 | 11.5 |
| user | 15.3 | 59.2 | 7.7 | 7.7 | 10.1 |
| cot | 6.6 | 9.6 | 55.4 | 19.6 | 8.8 |
| assistant | 8.3 | 10.7 | 20.0 | 51.9 | 9.0 |
| tool | 8.7 | 11.0 | 9.8 | 9.8 | 60.7 |

**Layer 8** (accuracy 75.4%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 59.8 | 22.6 | 6.5 | 2.6 | 8.5 |
| user | 10.4 | 82.8 | 2.2 | 1.1 | 3.5 |
| cot | 6.6 | 4.0 | 78.9 | 7.6 | 2.8 |
| assistant | 3.8 | 3.9 | 14.1 | 75.8 | 2.5 |
| tool | 9.2 | 4.9 | 3.9 | 2.2 | 79.8 |

**Layer 12** (accuracy 71.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 58.5 | 20.2 | 7.3 | 2.5 | 11.6 |
| user | 8.9 | 82.8 | 2.8 | 0.7 | 4.8 |
| cot | 8.8 | 5.0 | 69.3 | 11.2 | 5.7 |
| assistant | 4.2 | 2.7 | 14.0 | 75.7 | 3.4 |
| tool | 12.3 | 6.7 | 5.8 | 2.0 | 73.2 |

**Layer 16** (accuracy 81.8%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 72.5 | 10.3 | 6.1 | 1.8 | 9.4 |
| user | 4.0 | 92.8 | 1.4 | 0.1 | 1.7 |
| cot | 4.9 | 0.8 | 80.3 | 10.0 | 4.0 |
| assistant | 2.9 | 0.5 | 13.7 | 80.5 | 2.5 |
| tool | 7.6 | 1.6 | 5.5 | 2.2 | 83.2 |

**Layer 20** (accuracy 76.4%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 67.8 | 10.2 | 7.8 | 3.0 | 11.2 |
| user | 6.1 | 89.4 | 1.8 | 0.3 | 2.4 |
| cot | 7.4 | 0.9 | 72.2 | 13.6 | 5.9 |
| assistant | 3.1 | 0.7 | 16.7 | 76.0 | 3.6 |
| tool | 11.1 | 2.6 | 6.9 | 3.0 | 76.5 |

## Combined probe, C4/Dolma held-out


**Layer 4** (accuracy 43.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 45.8 | 25.4 | 5.4 | 5.8 | 17.5 |
| user | 27.9 | 50.6 | 4.3 | 4.5 | 12.7 |
| cot | 13.9 | 9.3 | 34.1 | 18.0 | 24.6 |
| assistant | 16.2 | 10.8 | 15.8 | 34.3 | 22.9 |
| tool | 13.6 | 9.6 | 11.4 | 10.9 | 54.5 |

**Layer 8** (accuracy 71.6%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 60.5 | 17.0 | 4.0 | 3.3 | 15.2 |
| user | 13.5 | 78.2 | 1.5 | 1.1 | 5.7 |
| cot | 12.1 | 0.9 | 62.4 | 9.8 | 14.8 |
| assistant | 1.8 | 0.2 | 5.8 | 90.8 | 1.4 |
| tool | 16.4 | 3.8 | 10.1 | 3.4 | 66.3 |

**Layer 12** (accuracy 64.2%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 55.0 | 12.5 | 5.0 | 6.4 | 21.2 |
| user | 14.8 | 69.4 | 2.7 | 1.9 | 11.1 |
| cot | 12.2 | 3.7 | 50.7 | 15.5 | 17.9 |
| assistant | 2.1 | 0.2 | 9.1 | 86.8 | 1.8 |
| tool | 18.3 | 7.3 | 9.8 | 5.8 | 58.9 |

**Layer 16** (accuracy 78.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 73.0 | 6.8 | 1.6 | 1.6 | 17.1 |
| user | 5.0 | 92.7 | 0.3 | 0.1 | 2.0 |
| cot | 3.8 | 0.5 | 68.4 | 8.4 | 19.0 |
| assistant | 0.1 | 0.1 | 7.7 | 91.6 | 0.5 |
| tool | 12.7 | 1.8 | 13.4 | 3.0 | 69.0 |

**Layer 20** (accuracy 70.6%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 63.7 | 7.3 | 4.7 | 3.4 | 20.8 |
| user | 6.7 | 90.4 | 0.7 | 0.1 | 2.2 |
| cot | 5.7 | 0.6 | 55.2 | 15.4 | 23.1 |
| assistant | 0.9 | 0.1 | 11.5 | 86.0 | 1.6 |
| tool | 16.9 | 2.3 | 17.3 | 5.8 | 57.7 |

## Combined probe, Toucan held-out


**Layer 4** (accuracy 53.6%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 41.8 | 27.4 | 9.1 | 10.5 | 11.2 |
| user | 14.5 | 59.9 | 7.3 | 8.6 | 9.7 |
| cot | 6.6 | 8.6 | 54.2 | 21.1 | 9.6 |
| assistant | 8.0 | 9.7 | 20.0 | 52.7 | 9.6 |
| tool | 8.9 | 10.0 | 10.2 | 11.4 | 59.5 |

**Layer 8** (accuracy 74.8%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 58.3 | 23.1 | 6.7 | 2.5 | 9.4 |
| user | 10.8 | 82.8 | 2.1 | 0.8 | 3.5 |
| cot | 5.2 | 3.6 | 79.2 | 8.9 | 3.1 |
| assistant | 3.2 | 3.8 | 13.3 | 76.7 | 2.9 |
| tool | 10.7 | 5.7 | 4.3 | 2.6 | 76.8 |

**Layer 12** (accuracy 70.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 56.9 | 20.2 | 8.3 | 2.6 | 11.9 |
| user | 8.7 | 83.1 | 2.8 | 0.6 | 4.9 |
| cot | 7.4 | 4.2 | 69.7 | 12.7 | 6.0 |
| assistant | 4.1 | 2.7 | 13.8 | 76.0 | 3.5 |
| tool | 15.5 | 6.8 | 6.5 | 2.2 | 68.9 |

**Layer 16** (accuracy 81.1%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 73.8 | 9.1 | 6.2 | 1.7 | 9.1 |
| user | 4.3 | 92.7 | 1.1 | 0.1 | 1.8 |
| cot | 4.3 | 0.6 | 79.8 | 11.0 | 4.3 |
| assistant | 3.0 | 0.6 | 12.8 | 80.5 | 3.0 |
| tool | 11.5 | 1.6 | 6.1 | 2.1 | 78.7 |

**Layer 20** (accuracy 75.3%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 67.8 | 9.5 | 6.7 | 3.0 | 13.0 |
| user | 5.9 | 89.4 | 1.5 | 0.3 | 3.0 |
| cot | 7.9 | 0.9 | 69.8 | 13.9 | 7.6 |
| assistant | 3.4 | 0.7 | 15.3 | 75.7 | 4.8 |
| tool | 14.5 | 2.4 | 6.5 | 3.0 | 73.6 |

## Clean chat token accuracy by layer (%)

| Arm | 4 | 8 | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|
| C4/Dolma | 48.2 | 91.9 | 87.2 | 79.1 | 80.7 |
| Toucan | 42.5 | 82.9 | 85.3 | 86.4 | 91.1 |
| Combined | 43.3 | 82.6 | 87.7 | 80.1 | 91.4 |

## AppWorld normal test: percentage confusion matrices


**C4/Dolma probe, layer 4** (accuracy 50.9%, macro recall 39.4%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 7.0 | 25.4 | 4.5 | 28.3 | 34.8 |
| user | 0.3 | 53.0 | 9.6 | 23.8 | 13.3 |
| cot | 0.1 | 2.0 | 44.8 | 5.3 | 47.9 |
| assistant | 0.1 | 0.2 | 9.6 | 2.3 | 87.8 |
| tool | 0.1 | 0.1 | 6.6 | 3.3 | 89.9 |

**C4/Dolma probe, layer 8** (accuracy 21.0%, macro recall 44.3%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.6 | 14.3 | 17.2 | 0.0 | 61.9 |
| user | 0.0 | 93.8 | 1.7 | 0.0 | 4.5 |
| cot | 0.1 | 2.2 | 92.7 | 2.3 | 2.8 |
| assistant | 0.4 | 0.5 | 86.0 | 9.6 | 3.5 |
| tool | 16.4 | 23.2 | 33.3 | 8.1 | 19.0 |

**C4/Dolma probe, layer 12** (accuracy 27.1%, macro recall 35.2%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.2 | 0.8 | 7.4 | 0.0 | 90.6 |
| user | 0.0 | 49.8 | 1.9 | 0.0 | 48.3 |
| cot | 0.0 | 3.5 | 74.7 | 15.2 | 6.6 |
| assistant | 0.2 | 0.6 | 60.4 | 36.9 | 1.9 |
| tool | 9.5 | 13.0 | 42.4 | 21.6 | 13.5 |

**C4/Dolma probe, layer 16** (accuracy 32.6%, macro recall 30.7%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.1 | 27.5 | 4.1 | 0.0 | 66.4 |
| user | 0.1 | 67.7 | 0.8 | 0.0 | 31.5 |
| cot | 0.9 | 0.4 | 18.1 | 7.4 | 73.2 |
| assistant | 3.6 | 0.2 | 20.6 | 23.9 | 51.7 |
| tool | 14.6 | 14.9 | 3.2 | 25.4 | 41.9 |

**C4/Dolma probe, layer 20** (accuracy 32.0%, macro recall 34.5%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.5 | 52.8 | 11.8 | 0.0 | 34.9 |
| user | 0.0 | 87.8 | 1.1 | 0.3 | 10.8 |
| cot | 1.2 | 7.0 | 23.0 | 6.5 | 62.2 |
| assistant | 2.6 | 8.1 | 22.0 | 13.1 | 54.2 |
| tool | 10.5 | 28.5 | 3.6 | 9.5 | 47.9 |

**Toucan probe, layer 4** (accuracy 53.5%, macro recall 45.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 32.0 | 41.0 | 7.8 | 5.3 | 13.9 |
| user | 1.1 | 62.5 | 2.4 | 10.0 | 24.0 |
| cot | 0.5 | 1.8 | 38.1 | 1.9 | 57.8 |
| assistant | 0.1 | 0.1 | 6.6 | 0.4 | 92.8 |
| tool | 0.1 | 0.2 | 2.7 | 0.6 | 96.3 |

**Toucan probe, layer 8** (accuracy 55.2%, macro recall 52.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 18.9 | 47.1 | 4.5 | 0.0 | 29.5 |
| user | 0.0 | 78.0 | 0.7 | 0.7 | 20.5 |
| cot | 0.2 | 1.1 | 69.8 | 2.0 | 26.9 |
| assistant | 1.0 | 0.4 | 31.2 | 9.6 | 57.8 |
| tool | 0.3 | 0.6 | 8.3 | 2.3 | 88.4 |

**Toucan probe, layer 12** (accuracy 52.5%, macro recall 45.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 10.2 | 6.1 | 4.1 | 0.4 | 79.1 |
| user | 0.0 | 45.7 | 0.6 | 0.4 | 53.3 |
| cot | 0.0 | 1.2 | 72.2 | 17.8 | 8.8 |
| assistant | 0.0 | 0.2 | 42.3 | 43.6 | 13.9 |
| tool | 0.1 | 0.5 | 15.6 | 25.8 | 57.9 |

**Toucan probe, layer 16** (accuracy 47.3%, macro recall 45.5%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 8.6 | 25.0 | 21.3 | 2.9 | 42.2 |
| user | 0.3 | 71.0 | 8.5 | 3.1 | 17.1 |
| cot | 0.1 | 0.0 | 58.3 | 11.6 | 30.0 |
| assistant | 0.4 | 0.0 | 24.9 | 29.4 | 45.2 |
| tool | 1.4 | 0.3 | 17.5 | 20.6 | 60.3 |

**Toucan probe, layer 20** (accuracy 55.1%, macro recall 46.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.5 | 35.7 | 14.3 | 3.3 | 44.3 |
| user | 0.1 | 72.1 | 8.3 | 5.7 | 13.7 |
| cot | 0.1 | 0.0 | 54.5 | 16.2 | 29.2 |
| assistant | 0.3 | 0.0 | 18.8 | 28.5 | 52.3 |
| tool | 0.6 | 0.1 | 10.2 | 12.4 | 76.7 |

**Combined probe, layer 4** (accuracy 52.6%, macro recall 44.0%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 19.3 | 45.9 | 8.2 | 8.6 | 18.0 |
| user | 0.6 | 60.3 | 4.6 | 15.9 | 18.6 |
| cot | 0.3 | 2.0 | 45.8 | 2.1 | 49.9 |
| assistant | 0.1 | 0.2 | 10.3 | 0.8 | 88.7 |
| tool | 0.2 | 0.2 | 4.9 | 1.0 | 93.7 |

**Combined probe, layer 8** (accuracy 38.8%, macro recall 52.1%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 18.4 | 43.4 | 4.9 | 0.0 | 33.2 |
| user | 0.0 | 88.8 | 0.6 | 0.3 | 10.2 |
| cot | 0.2 | 1.5 | 88.0 | 2.6 | 7.7 |
| assistant | 1.3 | 0.3 | 65.2 | 16.8 | 16.3 |
| tool | 2.7 | 3.6 | 28.7 | 16.6 | 48.5 |

**Combined probe, layer 12** (accuracy 38.1%, macro recall 40.3%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.9 | 4.1 | 0.8 | 0.4 | 91.8 |
| user | 0.0 | 46.7 | 0.8 | 0.1 | 52.4 |
| cot | 0.0 | 1.9 | 79.9 | 15.2 | 3.0 |
| assistant | 0.1 | 0.2 | 55.6 | 41.7 | 2.4 |
| tool | 0.7 | 1.2 | 29.8 | 37.8 | 30.5 |

**Combined probe, layer 16** (accuracy 41.1%, macro recall 40.9%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 4.1 | 26.7 | 10.3 | 2.5 | 56.5 |
| user | 0.1 | 71.0 | 5.0 | 0.7 | 23.3 |
| cot | 0.6 | 0.0 | 49.1 | 13.1 | 37.3 |
| assistant | 1.1 | 0.0 | 30.0 | 34.7 | 34.1 |
| tool | 3.6 | 0.8 | 18.2 | 31.9 | 45.6 |

**Combined probe, layer 20** (accuracy 52.3%, macro recall 44.0%)

| True \ Predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.8 | 28.2 | 14.7 | 2.0 | 54.2 |
| user | 0.0 | 68.4 | 4.4 | 3.0 | 24.2 |
| cot | 0.1 | 0.1 | 50.1 | 13.5 | 36.2 |
| assistant | 0.3 | 0.0 | 20.7 | 29.3 | 49.7 |
| tool | 1.0 | 0.0 | 10.3 | 17.2 | 71.5 |
