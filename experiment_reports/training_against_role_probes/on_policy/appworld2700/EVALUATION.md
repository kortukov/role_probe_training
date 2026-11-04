# Automatic final-checkpoint evaluation

Training dependency: `1816559`. Final checkpoint: `experiment_reports/training_against_role_probes/on_policy/appworld2700/runs/epoch1_recovery/checkpoints/step00169`.

Model export precedes fresh layer-12 probe fitting and independent benchmark collection. The probe uses 250 passages rendered in five roles (1,250 sequences), all content tokens, and the existing from-scratch cuML fitting recipe. Scoring replays whole conversations; histories above native context are explicitly excluded from role metrics while retained in task-success denominators.

| Benchmark | Tasks | Collection job | Scoring job | Result job |
| --- | ---: | --- | --- | --- |
| appworld_test | 168 | 1798692 | 1798696 | 1798697 |
| tau2_retail | 114 | 1798692 | 1798698 | 1798699 |
| swe_bench | 500 | 1798694 | 1798700 | 1798701 |

Each completed benchmark writes `post_training/results/<benchmark>.json` and `.md`. All jobs are submitted with success dependencies. SWE runs independently of AppWorld/retail and uses the current Inspect generous protocol, not a model-card reproduction. Queue availability still determines start times. Automatic GitHub publication is not part of these jobs.

[Exact submission records](evaluation_jobs.json). [Reproduction](RUNNING.md).
