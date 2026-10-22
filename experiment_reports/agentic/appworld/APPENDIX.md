# Supporting results

Task success counts all assigned attempts, including errors. Scenario success requires all three task variants to succeed and is omitted for incomplete scenario subsets. Step-limited attempts exhausted the Python interaction allowance.

| Dataset | Tasks | Successes | Errors | Step-limited | Task success | Scenario success |
|---|---:|---:|---:|---:|---:|---:|
| train | 90 | 38 | 0 | 3 | 42.2% | 23.3% |
| pilot_serial | 4 | 2 | 0 | 0 | 50.0% | — |
| pilot_one_gpu | 4 | 1 | 0 | 0 | 25.0% | — |
| pilot_two_gpus | 4 | 2 | 0 | 0 | 50.0% | — |

## Collection throughput

Collection time includes environment initialization, model calls, tool execution and scoring, but excludes inference-server startup and export. Parallel shards use the interval from the earliest start to the latest finish on the same node. Throughput counts attempts, not successes. Median rollout time excludes initialization and is available only for attempts reaching scoring.

| Dataset | GPUs | Concurrent worlds allowed | Collection minutes | Attempts/hour | Median rollout seconds |
|---|---:|---:|---:|---:|---:|
| train | 2 | 16 | 6.7 | 800.5 | 49.9 |
| pilot_serial | 1 | 1 | 2.9 | 84.1 | 40.6 |
| pilot_one_gpu | 1 | 8 | 1.8 | 132.3 | 50.7 |
| pilot_two_gpus | 2 | 16 | 1.9 | 126.2 | 46.9 |
