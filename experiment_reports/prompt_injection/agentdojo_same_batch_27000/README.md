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

[Automatically saved results and coverage](RESULTS.md) reproduce the historical
base: clean utility **66.32% (63/95)**, attacked utility **53.64% (494/921)**,
ASR **23.78% (219/921)** and ASR+ **24.00% (221/921)**, including two skips.
[Machine-readable metrics](metrics.json) retain counts and coverage.

At submission on 2026-11-03, the fresh base and trained evaluations were pending;
[RESULTS.md](RESULTS.md) records their latest automatically saved state. The historical base used a
different serving version and an `ipython` tool header; the current export uses
function-name tool headers. Its results provide historical context; the fresh,
matched base is the primary comparison. All runs use the historical 12,288-token
budget, below GPT-OSS's native context, so results describe this budgeted protocol.

## Message

The historical baseline is verified from saved trajectories. At submission, whether the trained
model improves robustness remained unanswered pending both new runs; reduced
attack success must be assessed together with clean and attacked utility.

[Reproduction, job records, model sharing and saved artifacts](RUNNING.md)
