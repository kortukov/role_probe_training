# GPT-OSS-20B on AppWorld

## Motivation & RQ

Can AppWorld provide useful on-policy agent trajectories, and how quickly can we collect a batch for future probe training?

## Evaluation setup

**Data and agent.** We collect AppWorld's 90 training tasks separately from its 168 normal-test and 417 challenge-test tasks. GPT-OSS-20B explores app documentation and executes Python through a stateful tool. A task finishes when the agent marks it complete or reaches an execution limit. Only training tasks are used to validate and tune the collection setup; test results are evaluated in aggregate. This experiment collects trajectories, without updating model or probe weights.

**Metrics and comparison.** Task success is the fraction of assigned tasks passing AppWorld's state-based evaluation, counting errors as failures. Training throughput is completed trajectory attempts per hour, including unsuccessful attempts. We compare sequential and concurrent worlds on the same seeded training subset using one inference GPU. Collection time excludes model startup; reproduction notes record startup separately. Test trajectories remain separate from future training data.

## Factual observations

Parallel execution passed two reference-solution checks on training tasks. The model throughput pilot is running; task performance and timing results are not yet available.

## Message

The environment and scoring path are validated on reference solutions. Whether this collection setup is fast enough for repeated on-policy training batches remains to be measured.

See [RUNNING.md](RUNNING.md) for collection commands, data locations and timing artifacts.
