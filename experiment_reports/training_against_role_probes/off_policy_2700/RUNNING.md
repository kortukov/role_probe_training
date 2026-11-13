# Running

Code: `src/experiments/training_against_role_probes/off_policy_2700/`; trainer
`src/role_probe_training/training/train_same_batch.py`. Run from the repository root with
the root environment (`uv sync --locked --extra vllm`). The job files are written for
DE.BER1; on another cluster change the `#SBATCH` header, the `local_job_dir.sh` line and
the cache paths as in the existing `hx2/` job files ([agent_docs/HX2.md](../../../agent_docs/HX2.md)).

```bash
E=src/experiments/training_against_role_probes/off_policy_2700
# 1. Rollouts: 30 array tasks of 90 training tasks (seeds 8301-8330) and one of 8 dev tasks.
sbatch $E/collect.sbatch
# 2. Validate and combine them, then build the same-batch KL masks.
sbatch --dependency=afterok:COLLECT $E/assemble.sbatch COLLECT
# 3. Two-update pilot, then the full run resumed from the pilot's checkpoint.
sbatch --time=02:00:00 --dependency=afterok:ASSEMBLE $E/train.sbatch $E/pilot_config.json
sbatch --dependency=afterok:PILOT $E/train.sbatch $E/train_config.json \
  --resume experiment_reports/training_against_role_probes/off_policy_2700/runs/training/epoch1/checkpoints/step00002
```

Settings (in `train_config.json` unless noted):

- Rollouts: base model in its MXFP4 checkpoint behind the Harmony proxy, temperature 1,
  high reasoning, 8,192 output tokens, 32,768 context, 50 steps, 8 concurrent tasks
  (`collect.sbatch`).
- Probe: `experiment_reports/gptoss_role_probes/lbfgs_fitter/probes/layer_16.npz`; for
  layer 12 change `probe` to `layer_12.npz`. The layer and activation site come from the
  probe file.
- Probe passages: `experiment_reports/gptoss_role_probes/lbfgs_fitter/artifacts/inputs/documents.jsonl`
  (250 passages; not in Git, written by the `prepare` stage of
  `src/experiments/gptoss_role_probes/run.py`).
- Learning rate 5e-5, KL coefficient 16, effective batch 16 on two GPUs (microbatch 2 per
  GPU), seed 123, one epoch. There is no refit step count: every refit runs L-BFGS to its
  gradient test and a refit that misses it stops training.

Outputs, under `experiment_reports/training_against_role_probes/off_policy_2700/` (ignored
by Git): `runs/collection/<array job>/shard<i>/`, `data/train2700/`, `data/dev/`,
`data/anchors.jsonl`, `runs/training/epoch1/` with checkpoints every 10 updates.

Checked without a GPU on 2026-11-13: `prepare.py` and the trainer's input checks on 168
existing AppWorld rollouts of the unified setup. Not yet run: collection through this job
file, the trainer with the L-BFGS refit on a GPU, and the evaluation of the trained model
(the old evaluation pipeline in `post_training/` has not been adapted to the Harmony proxy).
