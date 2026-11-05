# Role probes on No Robots: human conversations and model-generated turns

Status: complete. Date: 2026-10-30.

## 1. Motivation & RQ

The [training-data comparison](../../../gptoss_role_probes/training_data/README.md) fitted role probes on
neutral text, Toucan tool-use text, or both. How well do these three probe sets
recognise roles in ordinary instruction-following chat, both in human-written
multi-turn conversations and when GPT-OSS-20B produces its own reasoning and answer?

## 2. Evaluation setup

**Data.** The 500-sample No Robots test split (revision `e6f9a4ac`), 54 of which carry
a system message and 54 of which are multi-turn with up to five user turns.

**Version 1, as given.** Each conversation is rendered whole in Harmony: the sample's
own system message in the system turn when present, then the human user and assistant
turns, assistant turns as final-channel answers. No reasoning exists in this data.

**Version 2, generated.** The conversation is cut after the first user turn and placed
behind the GPT-OSS system turn that sets `Reasoning: medium`, followed by the sample's
system text when it has one. GPT-OSS-20B then samples its own analysis-channel
reasoning and final answer (vLLM, temperature 1, up to 4,000 tokens) from exactly the
Harmony prefix the probes later score. Generations that do not finish a final answer
are excluded and counted.

**Probes and metrics.** The C4/Dolma, Toucan and combined probes at layers 4 to 20,
reading pre-MLP activations of the frozen model, one forward pass per conversation.
Overall token accuracy weights content tokens equally; macro recall averages per-role
recall over roles present; per-turn accuracy is the token accuracy within each turn
position (system, user k, reasoning k, assistant k) pooled over conversations. The
uniform baseline is 20%; the majority baseline depends on the version's role mix.

## 3. Factual observations

Generation completed 489 of 500 prompts; 10 hit the 4,000-token cap without a final
answer and one answered on the commentary channel. Completed responses average about
1,180 tokens, roughly 1,950 characters of them reasoning. Full tables for all layers
are in [metrics/tables.md](metrics/tables.md); secondary analyses in
[metrics/analysis.json](metrics/analysis.json).

### Overall accuracy and macro recall at layer 12 (%)

| Probe | Version 1 accuracy | Version 1 macro recall | Version 2 accuracy | Version 2 macro recall |
|---|---:|---:|---:|---:|
| C4/Dolma | 82.9 | 65.4 | 78.8 | 61.2 |
| Toucan | 75.3 | 73.0 | 86.1 | 71.9 |
| Combined | 79.5 | 67.1 | 86.4 | 68.2 |

Accuracy is dominated by assistant tokens, which are 66% of version 1 and 52% of
version 2 content tokens; macro recall weights roles equally.

### Per-role recall at layer 12 (%)

| Probe | V1 system | V1 user | V1 assistant | V2 system | V2 user | V2 reasoning | V2 assistant |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4/Dolma | 46.2 | 51.0 | 99.1 | 6.8 | 80.2 | 57.9 | 99.9 |
| Toucan | 80.3 | 51.6 | 87.2 | 30.1 | 79.5 | 85.0 | 92.9 |
| Combined | 57.7 | 48.6 | 95.1 | 11.9 | 81.4 | 83.6 | 95.7 |

### Per-turn accuracy in version 1

| Probe, layer | system | user 1 | assistant 1 | user 2 | assistant 2 | user 3 | assistant 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4/Dolma, 12 | 46.2 | 51.4 | 99.1 | 47.9 | 99.5 | 33.5 | 98.8 |
| Toucan, 12 | 80.3 | 52.9 | 86.9 | 14.3 | 95.3 | 5.9 | 89.9 |
| Combined, 12 | 57.7 | 49.6 | 95.1 | 22.4 | 97.3 | 11.1 | 94.7 |
| C4/Dolma, 16 | 59.4 | 64.8 | 98.1 | 1.7 | 99.2 | 0.6 | 98.9 |
| Toucan, 16 | 92.3 | 57.6 | 88.9 | 0.1 | 95.8 | 0.0 | 94.9 |
| Combined, 16 | 77.6 | 57.4 | 90.1 | 0.3 | 95.4 | 0.0 | 93.4 |

Later user turns carry 697, 623, 73 and 18 tokens for turns 2 to 5; assistant turns
stay above 87% everywhere. User turns after the first are predicted as **assistant**:
at layer 12 the C4/Dolma probe labels 51% of their tokens assistant and 40% user, the
Toucan probe 88% assistant, and at layer 16 all three probes label 96% to 99% of them
assistant.

### Context effects on user and system turns

The same first user texts are recognised differently depending on what precedes them
(layer 12, token accuracy):

| Probe | V1, no system turn (446 samples) | V1, after the sample's system message (54) | V2, after the GPT-OSS preamble (489) |
|---|---:|---:|---:|
| C4/Dolma | 51.0 | 76.4 | 80.2 |
| Toucan | 52.7 | 67.8 | 79.5 |
| Combined | 49.3 | 66.9 | 81.4 |

The version 2 system turn is the fixed GPT-OSS preamble plus, for 54 samples, the
sample's instruction. Split by token position at layer 12, the preamble is recognised
at 7.0%, 31.0% and 12.2% by the three probes, and the instruction text that follows it
at 0.3%, 4.8% and 2.3%, although the same instructions standing alone as the system
turn in version 1 score 46%, 80% and 58%.

## 4. Message

On ordinary chat the three probe sets trade roles against each other rather than one
dominating. The neutral probe is near-perfect on human assistant answers and best on
overall accuracy in version 1, but weakest on system text and on the model's own
reasoning, where it labels 42% of reasoning tokens as something else. The Toucan probe
recognises real system prompts and generated reasoning far better, at a cost of 7 to
12 points on assistant answers, and the combined probe lands between them. By macro
recall the Toucan probe is best in both versions, by raw accuracy the neutral probe
wins version 1 and the Toucan and combined probes win version 2.

Two limitations hold for every probe set and matter more than the differences between
them. First, the probes do not track role across turns: a user turn that follows an
assistant answer is labelled assistant almost regardless of its content, so multi-turn
role recovery fails after the first exchange. Second, the same text is recognised
differently depending on what precedes it, with first user turns gaining about 30
points behind a system message and instruction text losing almost all recognition when
it follows the GPT-OSS preamble. Both are consistent with probes fitted only on
single-message wraps, which never show a role boundary inside a conversation. Training
on multi-turn wraps, or on histories with the native preamble, is the obvious next step
if conversational role recovery is the goal. Small-count turns (user 4 and 5, under 100
tokens) and the 54-sample system subset carry wide uncertainty.

[How to run this experiment](RUNNING.md)
