### Held-out token accuracy (%) by training arm and test collection

Uniform and majority baselines are 20.0 in every cell.

| Layer | C4/Dolma → C4/Dolma held-out | C4/Dolma → Toucan held-out | Toucan → C4/Dolma held-out | Toucan → Toucan held-out | Combined → C4/Dolma held-out | Combined → Toucan held-out |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 46.7 | 40.6 | 35.0 | 54.1 | 43.9 | 53.6 |
| 8 | 77.5 | 61.8 | 58.4 | 75.4 | 71.6 | 74.8 |
| 12 | 68.9 | 55.2 | 52.5 | 71.9 | 64.2 | 70.9 |
| 16 | 84.2 | 63.0 | 65.3 | 81.8 | 78.9 | 81.1 |
| 20 | 77.0 | 52.2 | 58.6 | 76.4 | 70.6 | 75.3 |

### Per-role recall (%) at layer 12

| Arm | Test collection | system | user | cot | assistant | tool |
|---|---|---:|---:|---:|---:|---:|
| C4/Dolma | C4/Dolma held-out | 63.9 | 74.7 | 55.8 | 88.1 | 62.0 |
| C4/Dolma | Toucan held-out | 41.4 | 71.7 | 47.9 | 72.4 | 42.8 |
| Toucan | C4/Dolma held-out | 38.1 | 62.5 | 51.6 | 84.6 | 25.5 |
| Toucan | Toucan held-out | 58.5 | 82.8 | 69.3 | 75.7 | 73.2 |
| Combined | C4/Dolma held-out | 55.0 | 69.4 | 50.7 | 86.8 | 58.9 |
| Combined | Toucan held-out | 56.9 | 83.1 | 69.7 | 76.0 | 68.9 |

### Per-role recall (%) at layer 16

| Arm | Test collection | system | user | cot | assistant | tool |
|---|---|---:|---:|---:|---:|---:|
| C4/Dolma | C4/Dolma held-out | 81.9 | 94.1 | 75.7 | 94.1 | 75.2 |
| C4/Dolma | Toucan held-out | 50.0 | 89.8 | 47.5 | 79.8 | 48.1 |
| Toucan | C4/Dolma held-out | 69.6 | 89.9 | 68.3 | 89.3 | 9.3 |
| Toucan | Toucan held-out | 72.5 | 92.8 | 80.3 | 80.5 | 83.2 |
| Combined | C4/Dolma held-out | 73.0 | 92.7 | 68.4 | 91.6 | 69.0 |
| Combined | Toucan held-out | 73.8 | 92.7 | 79.8 | 80.5 | 78.7 |

### Clean chat: conversation-balanced mean true-role probability (%), layer 12 / 16

| Arm | Role | Layer 12 | Layer 16 | Conversations |
|---|---|---:|---:|---:|
| C4/Dolma | system | 8.9 | 9.7 | 8 |
| C4/Dolma | user | 86.8 | 97.7 | 8 |
| C4/Dolma | cot | 63.5 | 40.1 | 8 |
| C4/Dolma | assistant | 99.5 | 97.5 | 8 |
| Toucan | system | 28.5 | 20.6 | 8 |
| Toucan | user | 86.8 | 95.4 | 8 |
| Toucan | cot | 85.3 | 85.2 | 8 |
| Toucan | assistant | 84.9 | 87.2 | 8 |
| Combined | system | 16.7 | 12.9 | 8 |
| Combined | user | 86.4 | 97.1 | 8 |
| Combined | cot | 85.7 | 75.0 | 8 |
| Combined | assistant | 89.1 | 85.8 | 8 |

### AppWorld normal test, whole histories: overall accuracy and per-role recall in percent

| Arm | Layer | Overall | Macro recall | system | user | cot | assistant | tool | Scored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4/Dolma | 4 | 50.9 | 39.4 | 7.0 | 53.0 | 44.8 | 2.3 | 89.9 | 168/168 |
| C4/Dolma | 8 | 21.0 | 44.3 | 6.6 | 93.8 | 92.7 | 9.6 | 19.0 | 168/168 |
| C4/Dolma | 12 | 27.1 | 35.2 | 1.2 | 49.8 | 74.7 | 36.9 | 13.5 | 168/168 |
| C4/Dolma | 16 | 32.6 | 30.7 | 2.1 | 67.7 | 18.1 | 23.9 | 41.9 | 168/168 |
| C4/Dolma | 20 | 32.0 | 34.5 | 0.5 | 87.8 | 23.0 | 13.1 | 47.9 | 168/168 |
| Toucan | 4 | 53.5 | 45.9 | 32.0 | 62.5 | 38.1 | 0.4 | 96.3 | 168/168 |
| Toucan | 8 | 55.2 | 52.9 | 18.9 | 78.0 | 69.8 | 9.6 | 88.4 | 168/168 |
| Toucan | 12 | 52.5 | 45.9 | 10.2 | 45.7 | 72.2 | 43.6 | 57.9 | 168/168 |
| Toucan | 16 | 47.3 | 45.5 | 8.6 | 71.0 | 58.3 | 29.4 | 60.3 | 168/168 |
| Toucan | 20 | 55.1 | 46.9 | 2.5 | 72.1 | 54.5 | 28.5 | 76.7 | 168/168 |
| Combined | 4 | 52.6 | 44.0 | 19.3 | 60.3 | 45.8 | 0.8 | 93.7 | 168/168 |
| Combined | 8 | 38.8 | 52.1 | 18.4 | 88.8 | 88.0 | 16.8 | 48.5 | 168/168 |
| Combined | 12 | 38.1 | 40.3 | 2.9 | 46.7 | 79.9 | 41.7 | 30.5 | 168/168 |
| Combined | 16 | 41.1 | 40.9 | 4.1 | 71.0 | 49.1 | 34.7 | 45.6 | 168/168 |
| Combined | 20 | 52.3 | 44.0 | 0.8 | 68.4 | 50.1 | 29.3 | 71.5 | 168/168 |

### Fit diagnostics

| Arm | Layer | Train tokens | L-BFGS iterations | Audit max abs gradient |
|---|---:|---:|---:|---:|
| C4/Dolma | 4 | 642530 | 266 | 1.6e-03 |
| C4/Dolma | 8 | 642530 | 429 | 1.6e-04 |
| C4/Dolma | 12 | 642530 | 451 | 9.0e-04 |
| C4/Dolma | 16 | 642530 | 662 | 9.0e-05 |
| C4/Dolma | 20 | 642530 | 849 | 2.0e-03 |
| Toucan | 4 | 1206190 | 281 | 8.3e-04 |
| Toucan | 8 | 1206190 | 287 | 1.0e-03 |
| Toucan | 12 | 1206190 | 359 | 1.2e-03 |
| Toucan | 16 | 1206190 | 459 | 4.8e-04 |
| Toucan | 20 | 1206190 | 738 | 9.3e-04 |
| Combined | 4 | 1848720 | 291 | 1.9e-03 |
| Combined | 8 | 1848720 | 328 | 9.9e-04 |
| Combined | 12 | 1848720 | 345 | 5.4e-04 |
| Combined | 16 | 1848720 | 386 | 7.2e-04 |
| Combined | 20 | 1848720 | 762 | 3.5e-03 |
