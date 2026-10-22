# GPT-OSS-20B on AppWorld

## Motivation & RQ

Can AppWorld provide useful on-policy agent trajectories, and how quickly can we collect a batch for future probe training?

## Evaluation setup

**Data and agent.** We collect AppWorld's 90 training tasks separately from its 168 normal-test and 417 challenge-test tasks. GPT-OSS-20B explores app documentation and executes Python through a stateful tool. A task finishes when the agent marks it complete or reaches an execution limit. Only training tasks are used to validate and tune the collection setup; test results are evaluated in aggregate. This experiment collects trajectories, without updating model or probe weights.

**Metrics and comparison.** Task success is the fraction of assigned tasks passing AppWorld's state-based evaluation, counting errors as failures. Training throughput is completed trajectory attempts per hour, including unsuccessful attempts. We compare sequential and concurrent collection of the same four seeded training tasks using one inference GPU, then repeat concurrent collection across two model replicas. Collection time excludes model startup; reproduction notes record startup separately. Test trajectories remain separate from future training data.

## Factual observations

The four-task training pilot completed without evaluation errors:

| Collection setup | Batch time |
|---|---:|
| One GPU, sequential worlds | 2.85 min |
| One GPU, concurrent worlds | 1.81 min |
| Two GPUs, concurrent worlds | 1.90 min |

The full training split completed in **6.75 minutes on two GPUs**: **38/90 tasks succeeded (42.2%)**, with no evaluation errors. Both held-out test splits are running separately. Pilot task scores and detailed timings are in the [appendix](APPENDIX.md). These are small, single-batch measurements; sampled trajectories differ across configurations.

## Message

Concurrent worlds reduce the measured four-trajectory batch time, but a second GPU does not improve this small batch. One inference GPU is therefore a reasonable starting point for small on-policy batches. Full collection uses two replicas to provide more capacity across many tasks; its throughput is measured separately.

These timings cover trajectory generation, not optimization or synchronization of updated model weights. Future on-policy training must keep each rollout batch tied to the learner's current checkpoint.

See [RUNNING.md](RUNNING.md) for collection commands, data locations and timing artifacts.
