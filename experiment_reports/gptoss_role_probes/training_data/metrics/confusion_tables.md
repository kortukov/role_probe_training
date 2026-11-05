# Confusion tables for the three probe sets

Each confusion table is row-normalized: for every true role, the percentage of its tokens
predicted as each role (rows sum to 100; the diagonal is the per-role recall). Token counts
per true role are in parentheses. Train and held-out sets wrap every document in all five
roles; AppWorld and No Robots carry their natural role mix, and No Robots has no tool tokens.


## Layer 12

**Overall token accuracy (%)**

| Probe | Train (own fitting data) | Held-out C4/Dolma | Held-out Toucan | AppWorld (168 histories) | No Robots v2 (489 generated first turns) |
|---|---:|---:|---:|---:|---:|
| C4/Dolma | 73.7 | 68.9 | 55.2 | 27.1 | 78.8 |
| Toucan | 73.7 | 52.5 | 71.9 | 52.5 | 86.1 |
| Combined | 69.4 | 64.2 | 70.9 | 38.1 | 86.4 |
| Num tokens | per probe, see below | 56,510 | 135,065 | 2,354,825 | 615,653 |

Train token counts differ per probe: C4/Dolma 642,530, Toucan 1,206,190, Combined 1,848,720.

### C4/Dolma probe, layer 12


**Train (own fitting data)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (128,506) | **65.1** | 12.6 | 6.5 | 2.1 | 13.7 |
| user (128,506) | 10.3 | **81.3** | 2.5 | 0.4 | 5.5 |
| reasoning (128,506) | 7.2 | 3.3 | **67.8** | 8.9 | 12.8 |
| assistant (128,506) | 1.7 | 0.3 | 7.7 | **88.3** | 2.0 |
| tool (128,506) | 13.6 | 6.2 | 11.9 | 2.0 | **66.2** |

**Held-out C4/Dolma**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (11,302) | **63.9** | 14.2 | 5.9 | 3.9 | 12.1 |
| user (11,302) | 16.7 | **74.7** | 2.0 | 0.7 | 5.9 |
| reasoning (11,302) | 11.7 | 3.2 | **55.8** | 13.0 | 16.4 |
| assistant (11,302) | 2.6 | 0.2 | 7.6 | **88.1** | 1.4 |
| tool (11,302) | 18.1 | 5.7 | 10.8 | 3.4 | **62.0** |

**Held-out Toucan**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (27,013) | **41.4** | 22.1 | 13.7 | 3.9 | 18.9 |
| user (27,013) | 11.7 | **71.7** | 4.1 | 0.9 | 11.6 |
| reasoning (27,013) | 10.7 | 7.1 | **47.9** | 21.4 | 13.0 |
| assistant (27,013) | 6.7 | 3.4 | 10.6 | **72.4** | 6.9 |
| tool (27,013) | 27.5 | 11.8 | 15.3 | 2.7 | **42.8** |

**AppWorld (168 histories)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (40,992) | **1.2** | 0.8 | 7.4 | 0.0 | 90.6 |
| user (13,117) | 0.0 | **49.8** | 1.9 | 0.0 | 48.3 |
| reasoning (173,846) | 0.0 | 3.5 | **74.7** | 15.2 | 6.6 |
| assistant (913,675) | 0.2 | 0.6 | 60.4 | **36.9** | 1.9 |
| tool (1,213,195) | 9.5 | 13.0 | 42.4 | 21.6 | **13.5** |

**No Robots v2 (489 generated first turns)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (28,778) | **6.8** | 28.4 | 5.5 | 0.0 | 59.3 |
| user (43,330) | 1.4 | **80.2** | 3.2 | 6.8 | 8.5 |
| reasoning (225,239) | 0.0 | 2.0 | **57.9** | 34.8 | 5.3 |
| assistant (318,306) | 0.0 | 0.0 | 0.1 | **99.9** | 0.0 |
| tool (0) | — | — | — | — | — |

### Toucan probe, layer 12


**Train (own fitting data)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (241,238) | **62.9** | 13.3 | 6.6 | 3.5 | 13.7 |
| user (241,238) | 9.5 | **83.0** | 1.6 | 0.8 | 5.1 |
| reasoning (241,238) | 6.5 | 2.2 | **71.4** | 14.7 | 5.3 |
| assistant (241,238) | 3.9 | 1.4 | 12.7 | **76.2** | 5.8 |
| tool (241,238) | 11.8 | 5.2 | 5.1 | 2.9 | **75.0** |

**Held-out C4/Dolma**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (11,302) | **38.1** | 15.2 | 10.9 | 20.6 | 15.3 |
| user (11,302) | 9.7 | **62.5** | 6.8 | 10.7 | 10.3 |
| reasoning (11,302) | 7.5 | 5.9 | **51.6** | 25.9 | 9.0 |
| assistant (11,302) | 1.0 | 0.5 | 12.5 | **84.6** | 1.4 |
| tool (11,302) | 16.3 | 13.1 | 22.6 | 22.4 | **25.5** |

**Held-out Toucan**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (27,013) | **58.5** | 20.2 | 7.3 | 2.5 | 11.6 |
| user (27,013) | 8.9 | **82.8** | 2.8 | 0.7 | 4.8 |
| reasoning (27,013) | 8.8 | 5.0 | **69.3** | 11.2 | 5.7 |
| assistant (27,013) | 4.2 | 2.7 | 14.0 | **75.7** | 3.4 |
| tool (27,013) | 12.3 | 6.7 | 5.8 | 2.0 | **73.2** |

**AppWorld (168 histories)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (40,992) | **10.2** | 6.1 | 4.1 | 0.4 | 79.1 |
| user (13,117) | 0.0 | **45.7** | 0.6 | 0.4 | 53.3 |
| reasoning (173,846) | 0.0 | 1.2 | **72.2** | 17.8 | 8.8 |
| assistant (913,675) | 0.0 | 0.2 | 42.3 | **43.6** | 13.9 |
| tool (1,213,195) | 0.1 | 0.5 | 15.6 | 25.8 | **57.9** |

**No Robots v2 (489 generated first turns)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (28,778) | **30.1** | 44.8 | 9.1 | 0.0 | 16.0 |
| user (43,330) | 2.3 | **79.5** | 5.8 | 8.9 | 3.5 |
| reasoning (225,239) | 0.0 | 1.1 | **85.0** | 13.7 | 0.1 |
| assistant (318,306) | 0.0 | 0.0 | 7.1 | **92.9** | 0.0 |
| tool (0) | — | — | — | — | — |

### Combined probe, layer 12


**Train (own fitting data)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (369,744) | **58.0** | 14.4 | 7.3 | 3.8 | 16.5 |
| user (369,744) | 10.3 | **79.8** | 2.3 | 0.9 | 6.7 |
| reasoning (369,744) | 7.5 | 3.3 | **65.4** | 14.4 | 9.4 |
| assistant (369,744) | 3.6 | 1.3 | 12.3 | **77.5** | 5.2 |
| tool (369,744) | 14.6 | 7.3 | 8.6 | 3.2 | **66.3** |

**Held-out C4/Dolma**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (11,302) | **55.0** | 12.5 | 5.0 | 6.4 | 21.2 |
| user (11,302) | 14.8 | **69.4** | 2.7 | 1.9 | 11.1 |
| reasoning (11,302) | 12.2 | 3.7 | **50.7** | 15.5 | 17.9 |
| assistant (11,302) | 2.1 | 0.2 | 9.1 | **86.8** | 1.8 |
| tool (11,302) | 18.3 | 7.3 | 9.8 | 5.8 | **58.9** |

**Held-out Toucan**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (27,013) | **56.9** | 20.2 | 8.3 | 2.6 | 11.9 |
| user (27,013) | 8.7 | **83.1** | 2.8 | 0.6 | 4.9 |
| reasoning (27,013) | 7.4 | 4.2 | **69.7** | 12.7 | 6.0 |
| assistant (27,013) | 4.1 | 2.7 | 13.8 | **76.0** | 3.5 |
| tool (27,013) | 15.5 | 6.8 | 6.5 | 2.2 | **68.9** |

**AppWorld (168 histories)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (40,992) | **2.9** | 4.1 | 0.8 | 0.4 | 91.8 |
| user (13,117) | 0.0 | **46.7** | 0.8 | 0.1 | 52.4 |
| reasoning (173,846) | 0.0 | 1.9 | **79.9** | 15.2 | 3.0 |
| assistant (913,675) | 0.1 | 0.2 | 55.6 | **41.7** | 2.4 |
| tool (1,213,195) | 0.7 | 1.2 | 29.8 | 37.8 | **30.5** |

**No Robots v2 (489 generated first turns)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (28,778) | **11.9** | 56.0 | 4.5 | 0.0 | 27.6 |
| user (43,330) | 2.1 | **81.4** | 4.5 | 6.3 | 5.6 |
| reasoning (225,239) | 0.0 | 1.3 | **83.6** | 14.8 | 0.3 |
| assistant (318,306) | 0.0 | 0.0 | 4.3 | **95.7** | 0.0 |
| tool (0) | — | — | — | — | — |

**Train, neutral part only**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (128,506) | **54.7** | 14.7 | 6.4 | 4.4 | 19.8 |
| user (128,506) | 10.9 | **76.6** | 3.1 | 1.3 | 8.1 |
| reasoning (128,506) | 9.5 | 4.7 | **59.1** | 11.7 | 15.1 |
| assistant (128,506) | 2.1 | 0.6 | 12.7 | **81.6** | 3.0 |
| tool (128,506) | 16.2 | 9.4 | 13.6 | 4.3 | **56.6** |

**Train, toucan part only**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (241,238) | **59.7** | 14.3 | 7.8 | 3.5 | 14.7 |
| user (241,238) | 9.9 | **81.5** | 1.9 | 0.6 | 6.0 |
| reasoning (241,238) | 6.4 | 2.6 | **68.7** | 15.8 | 6.4 |
| assistant (241,238) | 4.4 | 1.7 | 12.1 | **75.4** | 6.4 |
| tool (241,238) | 13.7 | 6.2 | 6.0 | 2.7 | **71.4** |

## Layer 16

**Overall token accuracy (%)**

| Probe | Train (own fitting data) | Held-out C4/Dolma | Held-out Toucan | AppWorld (168 histories) | No Robots v2 (489 generated first turns) |
|---|---:|---:|---:|---:|---:|
| C4/Dolma | 87.3 | 84.2 | 63.0 | 32.6 | 70.2 |
| Toucan | 83.7 | 65.3 | 81.8 | 47.3 | 84.2 |
| Combined | 80.5 | 78.9 | 81.1 | 41.1 | 80.4 |
| Num tokens | per probe, see below | 56,510 | 135,065 | 2,354,825 | 615,653 |

Train token counts differ per probe: C4/Dolma 642,530, Toucan 1,206,190, Combined 1,848,720.

### C4/Dolma probe, layer 16


**Train (own fitting data)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (128,506) | **85.3** | 3.3 | 2.3 | 0.5 | 8.7 |
| user (128,506) | 2.7 | **95.8** | 0.4 | 0.1 | 1.0 |
| reasoning (128,506) | 2.7 | 0.4 | **81.8** | 3.7 | 11.3 |
| assistant (128,506) | 0.2 | 0.0 | 3.3 | **95.6** | 0.8 |
| tool (128,506) | 9.8 | 1.5 | 10.2 | 0.8 | **77.8** |

**Held-out C4/Dolma**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (11,302) | **81.9** | 5.6 | 1.6 | 0.9 | 9.9 |
| user (11,302) | 4.9 | **94.1** | 0.1 | 0.0 | 0.9 |
| reasoning (11,302) | 3.1 | 0.4 | **75.7** | 6.8 | 13.9 |
| assistant (11,302) | 0.2 | 0.2 | 5.1 | **94.1** | 0.5 |
| tool (11,302) | 8.5 | 2.2 | 12.3 | 1.9 | **75.2** |

**Held-out Toucan**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (27,013) | **50.0** | 18.6 | 8.7 | 4.0 | 18.7 |
| user (27,013) | 3.8 | **89.8** | 1.7 | 0.2 | 4.4 |
| reasoning (27,013) | 7.0 | 4.3 | **47.5** | 26.0 | 15.1 |
| assistant (27,013) | 4.4 | 1.3 | 7.3 | **79.8** | 7.1 |
| tool (27,013) | 29.4 | 8.3 | 11.7 | 2.5 | **48.1** |

**AppWorld (168 histories)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (40,992) | **2.1** | 27.5 | 4.1 | 0.0 | 66.4 |
| user (13,117) | 0.1 | **67.7** | 0.8 | 0.0 | 31.5 |
| reasoning (173,846) | 0.9 | 0.4 | **18.1** | 7.4 | 73.2 |
| assistant (913,675) | 3.6 | 0.2 | 20.6 | **23.9** | 51.7 |
| tool (1,213,195) | 14.6 | 14.9 | 3.2 | 25.4 | **41.9** |

**No Robots v2 (489 generated first turns)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (28,778) | **11.5** | 66.4 | 1.7 | 0.0 | 20.4 |
| user (43,330) | 0.2 | **88.1** | 1.2 | 6.9 | 3.5 |
| reasoning (225,239) | 0.2 | 1.6 | **36.8** | 15.9 | 45.5 |
| assistant (318,306) | 0.0 | 0.1 | 0.0 | **96.7** | 3.1 |
| tool (0) | — | — | — | — | — |

### Toucan probe, layer 16


**Train (own fitting data)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (241,238) | **79.1** | 5.9 | 4.3 | 2.5 | 8.3 |
| user (241,238) | 4.0 | **94.5** | 0.4 | 0.2 | 0.9 |
| reasoning (241,238) | 4.1 | 0.6 | **79.8** | 11.5 | 4.0 |
| assistant (241,238) | 3.4 | 0.3 | 9.9 | **80.7** | 5.7 |
| tool (241,238) | 7.9 | 0.8 | 4.1 | 2.9 | **84.2** |

**Held-out C4/Dolma**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (11,302) | **69.6** | 12.0 | 6.4 | 10.2 | 1.9 |
| user (11,302) | 6.5 | **89.9** | 1.9 | 1.1 | 0.6 |
| reasoning (11,302) | 9.6 | 2.1 | **68.3** | 18.5 | 1.6 |
| assistant (11,302) | 0.5 | 0.1 | 10.1 | **89.3** | 0.0 |
| tool (11,302) | 31.5 | 5.5 | 33.0 | 20.7 | **9.3** |

**Held-out Toucan**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (27,013) | **72.5** | 10.3 | 6.1 | 1.8 | 9.4 |
| user (27,013) | 4.0 | **92.8** | 1.4 | 0.1 | 1.7 |
| reasoning (27,013) | 4.9 | 0.8 | **80.3** | 10.0 | 4.0 |
| assistant (27,013) | 2.9 | 0.5 | 13.7 | **80.5** | 2.5 |
| tool (27,013) | 7.6 | 1.6 | 5.5 | 2.2 | **83.2** |

**AppWorld (168 histories)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (40,992) | **8.6** | 25.0 | 21.3 | 2.9 | 42.2 |
| user (13,117) | 0.3 | **71.0** | 8.5 | 3.1 | 17.1 |
| reasoning (173,846) | 0.1 | 0.0 | **58.3** | 11.6 | 30.0 |
| assistant (913,675) | 0.4 | 0.0 | 24.9 | **29.4** | 45.2 |
| tool (1,213,195) | 1.4 | 0.3 | 17.5 | 20.6 | **60.3** |

**No Robots v2 (489 generated first turns)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (28,778) | **21.5** | 45.4 | 1.8 | 1.1 | 30.3 |
| user (43,330) | 1.4 | **79.9** | 9.5 | 7.1 | 2.1 |
| reasoning (225,239) | 0.1 | 0.1 | **86.2** | 12.1 | 1.6 |
| assistant (318,306) | 0.1 | 0.0 | 10.7 | **89.0** | 0.1 |
| tool (0) | — | — | — | — | — |

### Combined probe, layer 16


**Train (own fitting data)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (369,744) | **75.0** | 6.3 | 4.4 | 2.2 | 12.0 |
| user (369,744) | 4.5 | **93.2** | 0.6 | 0.2 | 1.5 |
| reasoning (369,744) | 4.7 | 0.8 | **75.5** | 10.7 | 8.3 |
| assistant (369,744) | 2.6 | 0.3 | 9.1 | **83.4** | 4.6 |
| tool (369,744) | 12.3 | 1.7 | 7.9 | 2.9 | **75.2** |

**Held-out C4/Dolma**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (11,302) | **73.0** | 6.8 | 1.6 | 1.6 | 17.1 |
| user (11,302) | 5.0 | **92.7** | 0.3 | 0.1 | 2.0 |
| reasoning (11,302) | 3.8 | 0.5 | **68.4** | 8.4 | 19.0 |
| assistant (11,302) | 0.1 | 0.1 | 7.7 | **91.6** | 0.5 |
| tool (11,302) | 12.7 | 1.8 | 13.4 | 3.0 | **69.0** |

**Held-out Toucan**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (27,013) | **73.8** | 9.1 | 6.2 | 1.7 | 9.1 |
| user (27,013) | 4.3 | **92.7** | 1.1 | 0.1 | 1.8 |
| reasoning (27,013) | 4.3 | 0.6 | **79.8** | 11.0 | 4.3 |
| assistant (27,013) | 3.0 | 0.6 | 12.8 | **80.5** | 3.0 |
| tool (27,013) | 11.5 | 1.6 | 6.1 | 2.1 | **78.7** |

**AppWorld (168 histories)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (40,992) | **4.1** | 26.7 | 10.3 | 2.5 | 56.5 |
| user (13,117) | 0.1 | **71.0** | 5.0 | 0.7 | 23.3 |
| reasoning (173,846) | 0.6 | 0.0 | **49.1** | 13.1 | 37.3 |
| assistant (913,675) | 1.1 | 0.0 | 30.0 | **34.7** | 34.1 |
| tool (1,213,195) | 3.6 | 0.8 | 18.2 | 31.9 | **45.6** |

**No Robots v2 (489 generated first turns)**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (28,778) | **10.2** | 56.3 | 3.4 | 0.0 | 30.1 |
| user (43,330) | 0.8 | **84.3** | 5.2 | 4.7 | 5.1 |
| reasoning (225,239) | 0.2 | 0.2 | **79.5** | 10.8 | 9.3 |
| assistant (318,306) | 0.7 | 0.1 | 11.4 | **86.9** | 1.0 |
| tool (0) | — | — | — | — | — |

**Train, neutral part only**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (128,506) | **74.9** | 5.7 | 3.0 | 1.3 | 15.1 |
| user (128,506) | 4.5 | **92.6** | 0.8 | 0.2 | 1.9 |
| reasoning (128,506) | 6.3 | 1.0 | **72.5** | 7.0 | 13.2 |
| assistant (128,506) | 0.6 | 0.1 | 8.1 | **89.8** | 1.4 |
| tool (128,506) | 15.7 | 2.6 | 13.5 | 2.9 | **65.3** |

**Train, toucan part only**

| True role (num tokens) | → system | → user | → reasoning | → assistant | → tool |
|---|---:|---:|---:|---:|---:|
| system (241,238) | **75.1** | 6.7 | 5.2 | 2.6 | 10.4 |
| user (241,238) | 4.5 | **93.5** | 0.5 | 0.2 | 1.3 |
| reasoning (241,238) | 3.9 | 0.7 | **77.2** | 12.6 | 5.6 |
| assistant (241,238) | 3.7 | 0.4 | 9.7 | **79.9** | 6.3 |
| tool (241,238) | 10.4 | 1.3 | 4.9 | 2.9 | **80.5** |
