# Running

Code: `src/experiments/training_against_role_probes/on_policy_progress/` (one JSON
configuration per run, `train.sbatch`, `evaluate.sbatch`, `results.py`); trainer
`src/role_probe_training/epoch_training/train.py`. Branch `on-policy-ladder`.

```bash
E=src/experiments/training_against_role_probes/on_policy_progress
R=experiment_reports/training_against_role_probes/on_policy_progress
# Validation rollouts: 8 dev tasks from the base model (array task 30 of the off-policy collection job).
DEV=$(sbatch --parsable --array=30 src/experiments/training_against_role_probes/off_policy_2700/collect.sbatch)
export DEV_SOURCE=experiment_reports/training_against_role_probes/off_policy_2700/runs/collection/$DEV/shard30/dev/trajectories.jsonl
# One run: train, then evaluate its last epoch checkpoint (epoch001 for small_k1, epoch002, epoch006).
T=$(sbatch --parsable --export=ALL --dependency=afterok:$DEV --job-name=ladder-small_k2 $E/train.sbatch small_k2)
sbatch --export=ALL --dependency=afterok:$T --job-name=eval-small_k2 $E/evaluate.sbatch small_k2 $R/small_k2/runs/training/epochs/epoch002
```

A configuration sets `epochs` (number of regenerations) and `attempts_per_task` (attempts on
each of the 90 tasks per regeneration). Outputs per run, under `<run>/` in the report folder
and ignored by Git: `runs/training/` (rollouts, checkpoints, logs per rank), `eval/`
(exported model, fresh probes, benchmark runs, per-token predictions) and `eval/results.json`.
The evaluation job needs the Azure credentials of the retail user simulator in its
environment.
