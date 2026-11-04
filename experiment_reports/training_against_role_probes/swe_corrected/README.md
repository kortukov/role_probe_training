# Corrected SWE-bench evaluation

> **Superseded:** The model-card investigation has stopped. The active comparison uses a [completed seven-model comparison](../swe_generous/README.md). Earlier attempts remain separate diagnostic artifacts.

## Motivation & RQ

How well do the base and five reported trained models complete SWE-bench Verified
tasks and recover conversational roles when the agent uses the correct repository
environment and saved patches are graded in fresh images?

## Evaluation setup

Evaluate all 500 Verified tasks once per model. Start with unchanged GPT-OSS-20B;
audit its complete results and remaining failure modes before collecting the
trained-model cohorts. The five trained models
are the original AppWorld, ASTRA and DepthBench models, plus the AppWorld agentic-KL
and same-batch-KL variants in the parent report.

Agent tools activate Conda `testbed`. The agent otherwise retains the Inspect
Python, persistent-shell and text-editor interface. A 2,000-message budget replaces
the previously binding 200-message cap. Generation uses BF16 weights, native 131K
context and high reasoning, with no cumulative-token cap and a two-hour task limit. The saved patch is applied to a separate fresh pinned
image and graded using the official SWE-bench 4.1.0 evaluation script and grader.
Apptainer supplies container orchestration. No outcome-selected retries are allowed.

Task success is successful tasks divided by all 500 assigned tasks, including
errors. Role accuracy is the fraction of labeled content tokens classified
correctly by each model's existing independently fitted standard-data layer-12
probe; per-role accuracy is recall. Replay every history whole. Histories above
131,072 tokens have no probe score, but keep their task outcome.

## Factual observations

The current corrected baseline solved **228/500 tasks (45.6%)**. All 500 attempts
have saved trajectories and scores. However, 33 Matplotlib grading runs crashed
during pytest initialization because `localhost` could not resolve. These were
incorrectly counted as task failures rather than evaluation errors. None hit
the task time, message or cumulative-token budgets; 20 reached the model's native
context bound. All 500 raw-log verdicts agree with the exported results.

Regrading the saved Matplotlib/Requests patches with localhost resolution repaired
recovers 15 successes, giving a provisional **243/500 (48.6%)** if all other scores
are retained. This measures a grading correction only. The original trajectories
remain intact, including agent-side environment failures.

The earlier interpreter controls passed, but a new Matplotlib gold-patch control
also fails with the same test-runner error. The five queued trained-model cohorts
are now held pending correction and broader controls. [Supporting audit](APPENDIX.md) and [pipeline state](pipeline.json).

## Message

The baseline is not yet valid for a matched comparison. Grading and agent-side
localhost failures must be corrected; the raw 45.6% is not a clean capability
estimate. Its score remains below the published 60.7% reference; the remaining gap is
unresolved, and this is not a reproduction of the model-card agent or inference
stack. Existing report rows retain the earlier setup until all six replacement
cohorts and probe results are complete and validated.

[Reproduction and continuation instructions](RUNNING.md)

Latest saved-patch diagnostics recover four more exact-hash successes (Astropy,
Sphinx and two Requests tasks), bringing the provisional total to **247/500 (49.4%)** while retaining all
other outcomes. This remains unvalidated as a corrected-agent comparison.
[Astropy/Sphinx regrade](base_audit_v2/repairs_regrade.json),
[Requests regrade](base_audit_v2/dual_http_regrade.json).
