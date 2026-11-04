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
