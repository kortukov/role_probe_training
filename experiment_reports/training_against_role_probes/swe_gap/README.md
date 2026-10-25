# Why is base GPT-OSS-20B below its reported SWE-bench score?

## Motivation & RQ

The matched native-context run solved 181/500 tasks (36.2%), versus 60.7% in
OpenAI's model card. Is this a scoring problem, an agent-environment problem,
or a difference in the agent interface and budgets?

## Evaluation setup

We audited all 500 saved base attempts without generating replacements. We also
replayed 21 saved patches in fresh pinned images using SWE-bench 4.1.0's official
`TestSpec.eval_script` and `get_eval_report`. Selection was fixed by hashed task ID:
one originally passing and one failing patch per repository where available.
This is a diagnostic sample, not an estimate of overall task success. Docker is
unavailable here; Apptainer supplies container orchestration, so this is not a
complete reproduction of the official Docker launcher.

Two fresh-image controls (Django and SymPy) compare direct execution, Inspect's
Python tool and its persistent shell, before and after activating the repository's
Conda environment. The audit counts traceback-style import-error lines in tool
output and attempts ending at the message limit; neither count by itself proves
why an individual task failed.

## Factual observations

| Check | Observation |
|---|---|
| Agent Python environment | All three execution paths initially use base Conda, not `testbed` |
| Required dependencies | Django's `asgiref` and SymPy's `mpmath` are unavailable initially and available after the fix |
| Saved attempts with traceback-style import errors | 383/500 |
| Attempts reaching 200-message cap | 190/500; 32 solved, 158 did not |
| Saved empty patches | 58/493 scored attempts; seven attempts have evaluation errors |
| Official-script patch regrading | 20/21 verdicts agree; one original pass becomes a failure |
| Requested reasoning | High on all 500 attempts |

The scorer activates `testbed` before testing, but the agent's tools do not.
Consequently the agent can see missing dependencies even though the grading
runtime has them. Saved logs include attempts to create replacement dependency
packages such as `mpmath` and `asgiref`. The new opt-in provider fixes the tool
environment; existing runs and scores have not been changed.

The one disagreement (`psf__requests-1142`) also fails when replayed with
Inspect’s script in a fresh image. Its original in-place success therefore does
not reproduce from the exported patch alone; the exact source of that state
difference remains unresolved. It is not evidence of an overly strict official grader.

The 200-message cap includes tool messages and is not 200 reasoning/action turns.
Only one model response hit its output-token limit, and the base cohort had no
native-context stops. More context alone therefore does not address these observed
constraints. Details and source provenance: [audit](audit.json),
[environment controls](environment_check.json), [patch checks](regrade.json).

## Message

**We found and reproduced a real environment bug.** It is a credible contributor
to poor performance, compounded by the message cap, but its effect on task success
still needs new controlled generations. The official-script check does not reveal
an undercount of successes; it also does not establish full scorer parity.

Changing from 500 to OpenAI's 477-task subset cannot explain the gap: even retaining
all 181 successes gives only 37.9%. Turning all seven evaluation errors into
successes would give 37.6% on 500 tasks. OpenAI describes a Codex-like terminal
interface, not the Inspect Python/shell/editor agent used here.
[Model card](https://deploymentsafety.openai.com/gpt-oss/evaluation).

An independent [HarmonyAgent paper](https://arxiv.org/html/2604.00362v1) reports
60.4% with native Harmony and tools such as `container.exec` and `apply_patch`.
Its public high-reasoning configuration permits 1,000 model steps, and the paper
says context-overflow runs were retried until completion. These differences make
it a useful implementation reference, not evidence that changing APIs alone will
recover 60% under our single-attempt policy. Our logs show only five attempts with
missing `apply_patch` command errors.

The next controlled comparison should first activate `testbed` while holding the
agent and budgets fixed, then separately test a larger message budget or native
Harmony tools. Keep new attempts separate and report all outcomes; do not replace
failed published attempts with successful retries.

[Reproduction instructions](RUNNING.md)
