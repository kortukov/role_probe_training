# Reproduction

Current results come from the complete base-model native-context cohort in
`experiment_reports/training_against_role_probes/base_model/`.
[Shared reproduction instructions](../../training_against_role_probes/RUNNING.md#corrected-swe-evaluation)
describe generation, scoring, environments and cluster artifact paths.
[Task summary](../../training_against_role_probes/base_model/post_training/swe_native_task_results.json)
records source paths and hashes; [pipeline provenance](../../training_against_role_probes/post_training/swe_native.json)
records settings and Slurm jobs.

This directory's pinned image manifest and image repairs remain shared inputs to
SWE jobs. Older `metrics/` and raw data describe the original constrained collection
and are not inputs to the current task or probe results. Current whole-history
probe summaries are linked from [the probe report](../probe_transfer/RUNNING.md).
