# Cross-evaluation of the three probe sets

Token accuracy in percent; macro recall is the mean of per-role recalls over roles present.
Train and held-out sets are role-balanced (chance 20%). AppWorld is the hx2 base cohort
(168 histories); No Robots is version 2 (489 first turns with generated reasoning and answer).


## Layer 12

| Probe | Train (own data) | Train C4/Dolma part | Train Toucan part | Held-out C4/Dolma | Held-out Toucan | AppWorld acc. | AppWorld macro | No Robots v2 acc. | No Robots v2 macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4/Dolma | 73.7 | 73.7 | — | 68.9 | 55.2 | 27.1 | 35.2 | 78.8 | 61.2 |
| Toucan | 73.7 | — | 73.7 | 52.5 | 71.9 | 52.5 | 45.9 | 86.1 | 71.9 |
| Combined | 69.4 | 65.7 | 71.4 | 64.2 | 70.9 | 38.1 | 40.3 | 86.4 | 68.2 |

**Per-role recall at layer 12** (AppWorld / No Robots v2)

| Probe | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| C4/Dolma | 1.2 / 6.8 | 49.8 / 80.2 | 74.7 / 57.9 | 36.9 / 99.9 | 13.5 / — |
| Toucan | 10.2 / 30.1 | 45.7 / 79.5 | 72.2 / 85.0 | 43.6 / 92.9 | 57.9 / — |
| Combined | 2.9 / 11.9 | 46.7 / 81.4 | 79.9 / 83.6 | 41.7 / 95.7 | 30.5 / — |

## Layer 16

| Probe | Train (own data) | Train C4/Dolma part | Train Toucan part | Held-out C4/Dolma | Held-out Toucan | AppWorld acc. | AppWorld macro | No Robots v2 acc. | No Robots v2 macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4/Dolma | 87.3 | 87.3 | — | 84.2 | 63.0 | 32.6 | 30.7 | 70.2 | 58.3 |
| Toucan | 83.7 | — | 83.7 | 65.3 | 81.8 | 47.3 | 45.5 | 84.2 | 69.1 |
| Combined | 80.5 | 79.0 | 81.2 | 78.9 | 81.1 | 41.1 | 40.9 | 80.4 | 65.2 |

**Per-role recall at layer 16** (AppWorld / No Robots v2)

| Probe | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| C4/Dolma | 2.1 / 11.5 | 67.7 / 88.1 | 18.1 / 36.8 | 23.9 / 96.7 | 41.9 / — |
| Toucan | 8.6 / 21.5 | 71.0 / 79.9 | 58.3 / 86.2 | 29.4 / 89.0 | 60.3 / — |
| Combined | 4.1 / 10.2 | 71.0 / 84.3 | 49.1 / 79.5 | 34.7 / 86.9 | 45.6 / — |
