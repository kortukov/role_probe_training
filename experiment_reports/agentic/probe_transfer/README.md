# Role-probe transfer on whole agent conversations

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
| AppWorld normal test | 12 | 26.54% | 1.23% | 49.94% | 73.14% | 35.39% | 13.32% |
| τ² retail | 12 | 19.95% | 6.64% | 29.89% | 20.03% | 94.95% | 0.69% |
| SWE-bench Verified | 12 | 36.22% | 0.93% | 92.48% | 42.82% | 79.67% | 2.28% |

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
