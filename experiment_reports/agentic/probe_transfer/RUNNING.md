# Reproduction

AppWorld/retail trajectories are reused; SWE uses the completed native-context cohort. Current whole-history
predictions are referenced in [metrics.json](metrics.json), under
`experiment_reports/training_against_role_probes/base_model/predictions/`.
The old chunked viewer/predictions are not the source of these results.

Run `src/experiments/training_against_role_probes/post_training/whole_score.sbatch`
with root `experiment_reports/training_against_role_probes/base_model`, `--base`,
and `--overlength exclude`. Retail output is explicitly selected with `--output`
pointing to `predictions/post_training_whole/tau2_retail`. Read
[the shared reproduction guide](../../training_against_role_probes/RUNNING.md#fresh-rollout-evaluation)
for environments, source paths and job provenance. No new model generations or
probe fits are needed for this baseline rerun.

Regenerate these summaries with:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/refresh_base_reports.py --native-swe
```

Inputs retain all recorded reasoning and tool messages in retrospective Harmony;
this is not a reconstruction of each generation-time prompt. The runtime uses
BF16 dequantization and flex attention. All metadata-labeled content tokens in
eligible histories are scored; synthetic tool schemas are excluded.

For SWE, use `post_training/score_swe_native.sbatch` with the same base root.
Its input is `base_model/data/swe_native/evaluation/trajectories.jsonl`; predictions
are in `base_model/predictions/swe_native/`. See the shared guide’s
[corrected SWE evaluation](../../training_against_role_probes/RUNNING.md#corrected-swe-evaluation)
for generation, task scoring and provenance.
