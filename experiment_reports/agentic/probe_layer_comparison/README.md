# Comparing role probes across layers on whole conversations

## Motivation & RQ

How well do neutral-text role probes recognize roles in GPT-OSS-20B agent histories?
This evaluation replaces the earlier chunked replay with whole-conversation scoring.

## Evaluation setup

We reuse 168 AppWorld normal-test trajectories and 114 τ² retail trajectories.
SWE uses 500 fresh unchanged-model attempts with matched BF16/native-context settings. Original task scores
and failed attempts are retained. Neutral C4/Dolma-trained linear probes read
post-attention-normalized pre-MLP states; message/channel metadata labels content.
Each scored conversation is processed in one forward pass. The one retail history
above the native 131,072-token limit is unscorable; no text is truncated or chunked.

Overall accuracy weights content tokens equally. Per-role accuracy is recall among
tokens of that true role. No probe fitting or selection uses these benchmarks.

## Factual observations

| Dataset | Layer | Overall | System | User | Reasoning | Assistant | Tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| AppWorld normal test | 4 | 51.26% | 6.97% | 53.21% | 46.13% | 2.20% | 90.27% |
| AppWorld normal test | 8 | 21.22% | 6.56% | 93.42% | 92.45% | 9.46% | 18.74% |
| AppWorld normal test | 12 | 26.54% | 1.23% | 49.94% | 73.14% | 35.39% | 13.32% |
| AppWorld normal test | 16 | 31.69% | 2.46% | 65.75% | 16.27% | 23.00% | 41.21% |
| AppWorld normal test | 20 | 31.29% | 0.82% | 89.53% | 23.43% | 12.42% | 47.04% |
| τ² retail | 4 | 12.35% | 0.34% | 9.47% | 6.25% | 8.84% | 97.91% |
| τ² retail | 8 | 55.94% | 23.68% | 23.97% | 62.02% | 91.58% | 3.66% |
| τ² retail | 12 | 19.95% | 6.64% | 29.89% | 20.03% | 94.95% | 0.69% |
| τ² retail | 16 | 33.06% | 9.26% | 6.43% | 35.56% | 97.81% | 4.11% |
| τ² retail | 20 | 65.39% | 1.34% | 19.12% | 74.31% | 92.28% | 13.33% |
| SWE-bench Verified | 4 | 50.99% | 1.39% | 61.96% | 6.58% | 0.16% | 96.12% |
| SWE-bench Verified | 8 | 23.90% | 4.63% | 93.26% | 78.78% | 39.20% | 2.43% |
| SWE-bench Verified | 12 | 36.22% | 0.93% | 92.48% | 42.82% | 79.67% | 2.28% |
| SWE-bench Verified | 16 | 29.24% | 3.24% | 91.03% | 34.01% | 53.12% | 9.92% |
| SWE-bench Verified | 20 | 29.81% | 0.00% | 88.02% | 24.01% | 36.89% | 24.75% |

Probe coverage is 168/168 AppWorld, 113/114 retail and 500/500 SWE histories.
[Percentage confusion matrices](APPENDIX.md) give the complete role breakdown.

## Message

Role recognition varies strongly by role and layer; no single aggregate accuracy
establishes reliable recognition of every role. These are base-model measurements,
not evidence about training effects. The corrected SWE run solves 181/500 tasks (36.2%); seven sandbox runtime
errors remain counted as failures. The current
[base-versus-trained comparison](../../training_against_role_probes/README.md)
reports task performance separately from role accuracy.

[Reproduction instructions](RUNNING.md)
