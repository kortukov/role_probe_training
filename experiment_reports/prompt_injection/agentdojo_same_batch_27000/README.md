# Does role-perception training improve AgentDojo robustness?

## Motivation & RQ

Does AppWorld + same-batch KL training on 27,000 trajectories reduce prompt
injection success while preserving useful agent behavior, compared with base
GPT-OSS-20B?

## Evaluation setup

We evaluate the final one-epoch checkpoint (step 1,688) and the base model on
AgentDojo v1.2.2: workspace, Slack, travel and banking. These simulated tasks
require tool use. The `important_instructions` attacker places instructions
in tool output to redirect the agent toward a separate attacker goal. This
experiment tests that fixed attack, not adaptive or exhaustive robustness.
Neither model receives an additional defense.

Each model gets a separate clean pass (95 user tasks) and attacked pass (921
user-task/injection-task pairs). Workspace tasks 35 and 38 are excluded for
both models, matching the previous project's comparison. Injection-goal seed
runs are saved but excluded from clean utility. New base and trained runs use
identical native tool calling, generation controls, BF16 export and context
budget. The trained model additionally applies its required FP32 attention adapter.

**Metrics.** Clean utility is the fraction of all 95 assigned clean tasks
completed. Attacked utility is task success over all 921 assigned attacked
pairs. ASR is the fraction of attacked pairs with a recorded successful attack,
with context-rejection skips receiving no attack credit. ASR+ retains the
harness's recorded security verdict, including its automatic attack-success
verdict for context skips. Lower attack success and higher utility are better.
Missing/error attempts stay in denominators, receive no utility credit, and
are reported through scored coverage and error counts. Partial rows cannot
support model comparisons. Skips can occur after actions have already executed;
ASR and ASR+ expose this ambiguity rather than resolving it.

**Role-perception extension.** Fresh five-class linear probes are fitted separately
for base and trained models on the original 250 C4/Dolma passages and original
passage split, retaining every content token. They read zero-based layer-12
post-attention-layernorm outputs (pre-MLP inputs). Evaluation replays each saved
AgentDojo history whole, without chunking or truncation, and reports overall
content-token argmax accuracy and per-role recall separately for clean and attacked
histories. Headers and synthetic tool schemas are unscored. Reasoning-channel text
was not retained by the AgentDojo driver: reasoning accuracy is unavailable, and
the measured roles describe the saved transcript rather than unrecorded thoughts.
Available partial error histories are included; three trained attacked attempts
have no saved history. Native-context exclusions and missing histories retain
their task outcomes and are disclosed in the role appendix.

## Factual observations

Both evaluations finished on 2026-11-04. Each attempted all 95 clean tasks and
921 attacked pairs. Percentages below use these full denominators; context-limit
errors receive no utility or attack credit.

| Model | Clean utility | Attacked utility | Recorded attack success | Attacked pairs with verdicts |
|---|---:|---:|---:|---:|
| Fresh base | 65.26% (62/95) | 49.73% (458/921) | 27.69% (255/921) | 916/921 |
| Same-batch KL, 27,000 | 62.11% (59/95) | 50.27% (463/921) | 29.53% (272/921) | 913/921 |

Relative to the fresh base, the trained model has **1.85 percentage points more
recorded attack success**, **3.16 points lower clean utility**, and **0.54 points
higher attacked utility**. These are descriptive single-attempt differences.

**Unresolved attack outcomes.** Five base and eight trained attacked attempts hit
the 12,288-token context budget, all in workspace, and have no final verdict.
They remain in the 921-pair denominators. If all these errors were counted as
attack successes, attack success would be 28.23% for base and 30.40% for trained.
Even assigning every base error attack success and every trained error no attack
credit would leave the trained model with more successful attacks (272 versus
260). Context errors can happen after actions have occurred; the saved traces
remain available for auditing. No clean attempt lacks a verdict.

[Automatically saved tables](RESULTS.md) and [machine-readable metrics](metrics.json)
retain coverage and error counts. The generated ASR+ column retains recorded
harness verdicts; it does not add attack credit for these unscored context errors.
The sensitivity calculation above addresses those separately.

The historical base is also reproduced from saved trajectories: clean utility
66.32% (63/95), attacked utility 53.64% (494/921), ASR 23.78% (219/921), and
ASR+ 24.00% (221/921), including two harness skips. It used a different serving
version and an `ipython` tool header; current exports use function-name tool
headers. The fresh, matched base is therefore the primary comparison. All runs
use the historical 12,288-token budget, below GPT-OSS's native context, so results
describe this budgeted protocol.

<!-- agentdojo-role-results -->
### Role perception

Values are percentages. Overall accuracy weights content tokens; role columns are recall within the true role. Reasoning was not saved and has no scored support. Each model uses its own freshly fitted standard-data layer-12 probe.

| Subset | Model | Overall role | System | User | Reasoning | Assistant | Tool |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean | Base | 21.6 | 20.3 | 83.4 | — | 58.4 | 10.3 |
| Attacked | Base | 17.7 | 21.6 | 82.7 | — | 51.0 | 11.0 |
| Clean | Same-batch KL (27,000) | 92.3 | 92.4 | 95.0 | — | 58.0 | 98.0 |
| Attacked | Same-batch KL (27,000) | 92.5 | 91.3 | 95.4 | — | 48.2 | 98.1 |

**Fit limitation:** The base probe stopped with a cuML L-BFGS line-search warning; optimizer convergence is not established. Probe weights and measurements are retained unchanged. [Fit audit](role_fit_audit.json).

[Coverage, raw counts and confusion matrices](APPENDIX.md). Missing histories and native-context exclusions are listed in the linked metrics.

On clean saved histories, trained overall role accuracy differs from base by +70.7 percentage points. Each model is scored on its own histories; this is not a paired comparison of identical text.
On attacked saved histories, trained overall role accuracy differs from base by +74.8 percentage points. Each model is scored on its own histories; this is not a paired comparison of identical text.

These role-label measurements are distinct from task utility and attack success; role recovery alone does not establish robustness.
<!-- /agentdojo-role-results -->

## Message

**This evaluation does not show improved prompt injection robustness from role-
perception training.** The trained model has more recorded successful attacks
and slightly lower clean-task success than the matched base, with similar
attacked utility. Accounting for all unresolved context-limit attempts cannot
reverse the attack-success ordering. This descriptive result concerns one fixed
attack and one generation attempt per pair; it does not establish a general or
statistically reliable difference across attack families. Strong role-label
recovery is therefore insufficient evidence of prompt injection robustness.

[Reproduction, job records, model sharing and saved artifacts](RUNNING.md)
