# Corrected SWE-bench evaluation

> **Superseded:** The model-card investigation has stopped. The active comparison uses a [matched stock Inspect protocol for all six models](../swe_standard/README.md). Earlier attempts remain separate diagnostic artifacts.

## Motivation & RQ

How well do the base and five reported trained models complete SWE-bench Verified
tasks and recover conversational roles when the agent uses the correct repository
environment and saved patches are graded in fresh images?

## Evaluation setup

Evaluate all 500 Verified tasks once per model. Start with unchanged GPT-OSS-20B;
review its complete results and remaining failure modes against the model-card
reference before collecting the trained-model cohorts. The five trained models
are the original AppWorld, ASTRA and DepthBench models, plus the AppWorld agentic-KL
and same-batch-KL variants in the parent report.

Agent tools activate Conda `testbed`. The agent otherwise retains the Inspect
Python, persistent-shell and text-editor interface. A 2,000-message budget replaces
the previously binding 200-message cap. Generation uses BF16 weights, native 131K
context and high reasoning. The saved patch is applied to a separate fresh pinned
image and graded using the official SWE-bench 4.1.0 evaluation script and grader.
Apptainer supplies container orchestration. No outcome-selected retries are allowed.

Task success is successful tasks divided by all 500 assigned tasks, including
errors. Role accuracy is the fraction of labeled content tokens classified
correctly by each model's existing independently fitted standard-data layer-12
probe; per-role accuracy is recall. Replay every history whole. Histories above
131,072 tokens have no probe score, but keep their task outcome.

## Factual observations

Collection is in progress; no new task or role accuracy is reported yet.
The corrected scorer passed gold-patch and unchanged-repository controls on Django
and SymPy. [Pipeline state and model scope](pipeline.json) record the current jobs.

## Message

The known environment bug and message-budget constraint are addressed in this
new setup. Their effect on task success remains to be measured. The published
60.7% is a diagnostic reference, not a score to obtain by replacing failed attempts
or selecting favorable runs. Existing report rows remain labeled with the earlier
setup until all replacement cohorts and probe results are complete and validated.

[Reproduction and continuation instructions](RUNNING.md)
