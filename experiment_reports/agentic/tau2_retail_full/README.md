# GPT-OSS-20B on full τ²-bench retail

## Motivation & RQ

How often does GPT-OSS-20B complete retail customer-service tasks, and do its conversations provide realistic data for role-probe evaluation?

## Evaluation setup

**Tasks and agent.** We evaluate all 114 τ²-bench retail tasks once. GPT-OSS-20B acts as the retail assistant, interacting with a GPT-4.1 customer simulator and using retail tools. The benchmark checks the resulting database state and required communicated information. Successful and unsuccessful histories are both retained.

**Metrics.** Task success is the fraction of all selected tasks passed; errors and interrupted attempts are not excluded. A 95% Wilson confidence interval describes uncertainty in this one-attempt success rate. We also count histories containing tool replies to establish their usefulness for probe evaluation.

## Factual observations

| Tasks solved | Task success | 95% confidence interval | Histories containing tool replies |
|---|---:|---|---:|
| 73 / 114 | 64.04% | 54.90%–72.25% | 112 / 114 |

Four attempts reached an execution limit. Although no task ended with a top-level evaluation error, recoverable model-call failures and output truncation occurred. [Additional diagnostics](APPENDIX.md) describe these limitations. The histories were subsequently used in the [probe-transfer experiment](../probe_transfer/README.md).

## Message

The run produces a complete, scored set of realistic retail histories. The 64.04% result is contextual evidence of task competence, not a reproduction of the published 54.8% result: that reference uses the original τ-bench, whereas this experiment uses τ²-bench. [OpenAI model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations).

Probe recognition and task success answer different questions. The retained failed and interrupted attempts are part of the probe evaluation, rather than selecting only successful conversations.

See [RUNNING.md](RUNNING.md) for execution settings, data and viewer reproduction.
