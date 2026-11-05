# Checkpoint role perception

Rows are true roles and columns are probe predictions. Counts include only content tokens. Each percentage row divides by its true-role token count. Headers and synthetic schemas are excluded. Role coverage excludes whole histories above native context; task success retains every assigned attempt.

## AppWorld normal test, epoch 1

Task success: 62/168. Role coverage: 168/168 histories. Native context: 131,072 tokens. [Counts and provenance](evaluations/epoch001/post_training/results/appworld_test.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 40992 | 0 | 0 | 0 | 0 |
| User | 3 | 13108 | 5 | 1 | 0 |
| Reasoning | 2 | 1157 | 176253 | 3653 | 2064 |
| Assistant | 0 | 25 | 1121 | 783982 | 2769 |
| Tool | 1 | 26 | 43 | 5672 | 1252594 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| User | 0.0 | 99.9 | 0.0 | 0.0 | 0.0 |
| Reasoning | 0.0 | 0.6 | 96.2 | 2.0 | 1.1 |
| Assistant | 0.0 | 0.0 | 0.1 | 99.5 | 0.4 |
| Tool | 0.0 | 0.0 | 0.0 | 0.5 | 99.5 |

Mean probability assigned to the true role, averaged across tokens of that role (distinct from argmax recall):

| System | User | Reasoning | Assistant | Tool |
|---:|---:|---:|---:|---:|
| 99.95 | 99.83 | 95.09 | 98.97 | 99.01 |

Excluded histories: 0; identities and lengths are in the linked result.

## τ² retail, epoch 1

Task success: 62/114. Role coverage: 114/114 histories. Native context: 131,072 tokens. [Counts and provenance](evaluations/epoch001/post_training/results/tau2_retail.json).

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 32946 | 103284 | 31578 | 0 | 2166 |
| User | 2051 | 11586 | 1235 | 10851 | 662 |
| Reasoning | 15347 | 484 | 1276049 | 80423 | 68 |
| Assistant | 1929 | 18 | 88927 | 309075 | 842 |
| Tool | 420 | 885 | 35126 | 85382 | 81929 |

Row percentages:

| True / predicted | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|
| System | 19.4 | 60.8 | 18.6 | 0.0 | 1.3 |
| User | 7.8 | 43.9 | 4.7 | 41.1 | 2.5 |
| Reasoning | 1.1 | 0.0 | 93.0 | 5.9 | 0.0 |
| Assistant | 0.5 | 0.0 | 22.2 | 77.1 | 0.2 |
| Tool | 0.2 | 0.4 | 17.2 | 41.9 | 40.2 |

Mean probability assigned to the true role, averaged across tokens of that role (distinct from argmax recall):

| System | User | Reasoning | Assistant | Tool |
|---:|---:|---:|---:|---:|
| 20.79 | 43.74 | 91.84 | 76.56 | 40.57 |

Excluded histories: 0; identities and lengths are in the linked result.
