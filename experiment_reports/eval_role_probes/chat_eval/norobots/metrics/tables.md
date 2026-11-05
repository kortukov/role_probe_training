### Version 2 generation

| Prompts | Completed | Failed | Mean completion tokens | Mean reasoning characters (completed) |
|---:|---:|---:|---:|---:|
| 500 | 489 | 11 | 1182 | 1949 |

Failure reasons: length: Generation did not finish a final answer (10); stop: Unexpected generated channel: commentary (1)

### Version 1: conversations as given (500 conversations)

Overall token accuracy / macro recall (%) by layer.

| Arm | 4 | 8 | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|
| C4/Dolma | 45.8 / 49.1 | 88.1 / 64.6 | 82.9 / 65.4 | 86.2 / 73.5 | 90.4 / 77.7 |
| Toucan | 42.4 / 57.9 | 84.2 / 82.2 | 75.3 / 73.0 | 78.1 / 79.1 | 80.9 / 76.4 |
| Combined | 39.9 / 52.4 | 84.3 / 70.8 | 79.5 / 67.1 | 78.7 / 74.5 | 83.8 / 75.1 |

**Per-role recall (%) at layer 12**

| Arm | system | user | assistant |
|---|---:|---:|---:|
| C4/Dolma | 46.2 | 51.0 | 99.1 |
| Toucan | 80.3 | 51.6 | 87.2 |
| Combined | 57.7 | 48.6 | 95.1 |

**Per-turn token accuracy (%) at layer 12** (tokens in the turn across conversations in parentheses)

| Arm | system | user 1 | assistant 1 | user 2 | assistant 2 | user 3 | assistant 3 | user 4 | assistant 4 | user 5 | assistant 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4/Dolma | 46.2 (905) | 51.4 (43962) | 99.1 (87320) | 47.9 (697) | 99.5 (1588) | 33.5 (623) | 98.8 (1737) | 24.7 (73) | 100.0 (263) | 27.8 (18) | 100.0 (105) |
| Toucan | 80.3 (905) | 52.9 (43962) | 86.9 (87320) | 14.3 (697) | 95.3 (1588) | 5.9 (623) | 89.9 (1737) | 1.4 (73) | 88.2 (263) | 0.0 (18) | 99.0 (105) |
| Combined | 57.7 (905) | 49.6 (43962) | 95.1 (87320) | 22.4 (697) | 97.3 (1588) | 11.1 (623) | 94.7 (1737) | 5.5 (73) | 97.0 (263) | 0.0 (18) | 100.0 (105) |

**Per-role recall (%) at layer 16**

| Arm | system | user | assistant |
|---|---:|---:|---:|
| C4/Dolma | 59.4 | 62.9 | 98.1 |
| Toucan | 92.3 | 55.8 | 89.1 |
| Combined | 77.6 | 55.6 | 90.3 |

**Per-turn token accuracy (%) at layer 16** (tokens in the turn across conversations in parentheses)

| Arm | system | user 1 | assistant 1 | user 2 | assistant 2 | user 3 | assistant 3 | user 4 | assistant 4 | user 5 | assistant 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4/Dolma | 59.4 (905) | 64.8 (43962) | 98.1 (87320) | 1.7 (697) | 99.2 (1588) | 0.6 (623) | 98.9 (1737) | 0.0 (73) | 100.0 (263) | 0.0 (18) | 100.0 (105) |
| Toucan | 92.3 (905) | 57.6 (43962) | 88.9 (87320) | 0.1 (697) | 95.8 (1588) | 0.0 (623) | 94.9 (1737) | 0.0 (73) | 89.7 (263) | 0.0 (18) | 98.1 (105) |
| Combined | 77.6 (905) | 57.4 (43962) | 90.1 (87320) | 0.3 (697) | 95.4 (1588) | 0.0 (623) | 93.4 (1737) | 0.0 (73) | 93.2 (263) | 0.0 (18) | 98.1 (105) |

### Version 2: first turn with generated reasoning and answer (489 conversations)

Overall token accuracy / macro recall (%) by layer.

| Arm | 4 | 8 | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|
| C4/Dolma | 52.2 / 53.2 | 88.7 / 72.1 | 78.8 / 61.2 | 70.2 / 58.3 | 69.8 / 58.1 |
| Toucan | 47.7 / 53.6 | 87.4 / 77.2 | 86.1 / 71.9 | 84.2 / 69.1 | 83.6 / 67.6 |
| Combined | 48.0 / 55.5 | 88.1 / 76.7 | 86.4 / 68.2 | 80.4 / 65.2 | 84.4 / 71.1 |

**Per-role recall (%) at layer 12**

| Arm | system | user | cot | assistant |
|---|---:|---:|---:|---:|
| C4/Dolma | 6.8 | 80.2 | 57.9 | 99.9 |
| Toucan | 30.1 | 79.5 | 85.0 | 92.9 |
| Combined | 11.9 | 81.4 | 83.6 | 95.7 |

**Per-turn token accuracy (%) at layer 12** (tokens in the turn across conversations in parentheses)

| Arm | system | user 1 | reasoning 1 | assistant 1 |
|---|---:|---:|---:|---:|
| C4/Dolma | 6.8 (28778) | 80.2 (43330) | 57.9 (225239) | 99.9 (318306) |
| Toucan | 30.1 (28778) | 79.5 (43330) | 85.0 (225239) | 92.9 (318306) |
| Combined | 11.9 (28778) | 81.4 (43330) | 83.6 (225239) | 95.7 (318306) |

**System-turn accuracy (%) at layer 12 by whether the sample supplied a system message**

| Arm | With sample system text | Without |
|---|---:|---:|
| C4/Dolma | 5.5 (3983) | 7.0 (24795) |
| Toucan | 25.0 (3983) | 30.9 (24795) |
| Combined | 10.0 (3983) | 12.2 (24795) |

**Per-role recall (%) at layer 16**

| Arm | system | user | cot | assistant |
|---|---:|---:|---:|---:|
| C4/Dolma | 11.5 | 88.1 | 36.8 | 96.7 |
| Toucan | 21.5 | 79.9 | 86.2 | 89.0 |
| Combined | 10.2 | 84.3 | 79.5 | 86.9 |

**Per-turn token accuracy (%) at layer 16** (tokens in the turn across conversations in parentheses)

| Arm | system | user 1 | reasoning 1 | assistant 1 |
|---|---:|---:|---:|---:|
| C4/Dolma | 11.5 (28778) | 88.1 (43330) | 36.8 (225239) | 96.7 (318306) |
| Toucan | 21.5 (28778) | 79.9 (43330) | 86.2 (225239) | 89.0 (318306) |
| Combined | 10.2 (28778) | 84.3 (43330) | 79.5 (225239) | 86.9 (318306) |

**System-turn accuracy (%) at layer 16 by whether the sample supplied a system message**

| Arm | With sample system text | Without |
|---|---:|---:|
| C4/Dolma | 9.1 (3983) | 11.9 (24795) |
| Toucan | 17.3 (3983) | 22.1 (24795) |
| Combined | 8.1 (3983) | 10.5 (24795) |
