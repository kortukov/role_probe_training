# Automatic final-checkpoint evaluation

Training dependency: `1816551`. Final checkpoint: `experiment_reports/training_against_role_probes/on_policy/appworld270_batch16/runs/epoch1_recovery/checkpoints/step00017`.

Model export precedes fresh layer-12 probe fitting and independent benchmark collection. The probe uses 250 passages rendered in five roles (1,250 sequences), all content tokens, and the existing from-scratch cuML fitting recipe. Scoring replays whole conversations; histories above native context are explicitly excluded from role metrics while retained in task-success denominators.

| Benchmark | Tasks | Collection job | Scoring job | Result job |
| --- | ---: | --- | --- | --- |
| appworld_test | 168 | 1798680 | 1798684 | 1798685 |
| tau2_retail | 114 | 1798680 | 1798686 | 1798687 |
| swe_bench | 500 | 1798682 | 1798688 | 1798689 |

Each completed benchmark writes `post_training/results/<benchmark>.json` and `.md`. All jobs are submitted with success dependencies. SWE runs independently of AppWorld/retail and uses the current Inspect generous protocol, not a model-card reproduction. Queue availability still determines start times. Automatic GitHub publication is not part of these jobs.

[Exact submission records](evaluation_jobs.json). [Reproduction](RUNNING.md).
