# GPT-OSS-20B on AppWorld

## Motivation & RQ

Can AppWorld provide useful on-policy agent trajectories, and how quickly can we collect a batch for future probe training?

## Evaluation setup

**Data and agent.** We collect AppWorld's 90 training tasks separately from its 168 normal-test and 417 challenge-test tasks. GPT-OSS-20B explores app documentation and executes Python through a stateful tool. A trajectory ends when the agent stops, marks the task complete or reaches an execution limit. Only training tasks are used to validate and tune the collection setup; test results are evaluated in aggregate. This experiment collects trajectories, without updating model or probe weights.

**Role-probe evaluation.** On the 168 normal-test trajectories, we also apply frozen
linear probes trained on neutral role-labeled text. The main probe reads layer 12's
pre-MLP activations and predicts five conversational roles from each content token;
metadata supplies the labels. Token accuracy weights tokens equally, while mean
trajectory accuracy weights conversations equally. Role recall is the fraction of
a role's tokens classified correctly. The majority-role baseline always predicts
tool, the most common role. Long histories are replayed in separate segments,
so later segments lack earlier context. No fitting uses the test split.

**Metrics and comparison.** Task success is the fraction of assigned tasks passing AppWorld's state-based evaluation, counting errors as failures. Training throughput is completed trajectory attempts per hour, including unsuccessful attempts. We compare sequential and concurrent collection of the same four seeded training tasks using one inference GPU, then repeat concurrent collection across two model replicas. Collection time excludes model startup; reproduction notes record startup separately. Test trajectories remain separate from future training data.

## Factual observations

All 675 tasks completed without evaluation errors:

| Split | Tasks solved | Task success | Collection time (two GPUs) |
|---|---:|---:|---:|
| Train | 38 / 90 | 42.2% | 6.75 min |
| Normal test | 45 / 168 | 26.8% | 14.39 min |
| Challenge test | 105 / 417 | 25.2% | 38.96 min |

The four-task training pilot completed without evaluation errors:

| Collection setup | Batch time |
|---|---:|
| One GPU, sequential worlds | 2.85 min |
| One GPU, concurrent worlds | 1.81 min |
| Two GPUs, concurrent worlds | 1.90 min |

All trajectories preserve explicit assistant reasoning; 666 also contain tool replies. Unsuccessful and execution-limited attempts remain in the corpus. Pilot task scores, execution limits and detailed timings are in the [appendix](APPENDIX.md). The pilot comparisons are small, single-batch measurements; sampled trajectories differ across configurations.

The layer 12 role probe scores **23.94% token accuracy**, below the **51.34%**
majority-role baseline. Mean trajectory accuracy is 24.70%, and tool recall is 14.01%.
All 168 trajectories and 2,364,547 content tokens were scored. [Per-role confusion
matrices and the five-layer comparison](../appworld_probe_transfer/APPENDIX.md)
provide the detailed breakdown; [the probe report](../appworld_probe_transfer/README.md)
explains the replay protocol and limitations.

## Message

Concurrent worlds reduce the measured four-trajectory batch time, but a second GPU did not reduce latency in this small pilot. One inference GPU is therefore a reasonable starting point for small on-policy batches. Full collection uses two replicas to provide more capacity across many tasks; its throughput is measured separately.

AppWorld provides a complete corpus of reasoning-bearing agent trajectories, with successful and unsuccessful attempts in each split. Training tasks have higher measured success than either test split; test data remains held out from future probe training.

Role recognition remains weak before AppWorld-based model training, especially for
tool messages; benchmark task success and role-probe accuracy measure different things.

These timings cover trajectory generation, not optimization or synchronization of updated model weights. Future on-policy training must keep each rollout batch tied to the learner's current checkpoint.

See [RUNNING.md](RUNNING.md) for collection commands, data locations and timing artifacts.
