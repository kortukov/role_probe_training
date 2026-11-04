# Reproduction

The 27,000-trajectory experiment retains the original 2,700 training attempts
(seeds 8301–8330) and adds seeds 8331–8600 on each of the same 90 AppWorld
training tasks. No outcome filtering is applied. Evaluation tasks are excluded.
Generation uses the original base model and settings: native MXFP4, temperature
1, high reasoning, 8,192 output tokens, 32,768 context, 50 steps, eight concurrent
worlds per server. New jobs use private compile caches.

Code: `src/experiments/training_against_role_probes/data_iteration/`.
Large artifacts stay under this experiment directory on cluster storage.
Training starts from the base model and keeps the same-batch KL settings:
one epoch, effective batch 16, learning rate 5e-5, KL coefficient 16, rank-64
LoRA, full histories, and standard 250-passage probe refits. The larger corpus
requires 1,688 updates (last batch eight), compared with 169 previously.
Thus data volume and training compute both increase. AppWorld normal test and
retail used fresh model-generated histories and fresh standard-data probes;
SWE remains deferred to the separate evaluation effort.

## Completed pipeline (2026-10-28)

Job IDs are recorded in [jobs.json](jobs.json). Collection array 1505943
uses indices 30–299 and seeds 8331–8600. It started with a single active
shard; after worker initialization and successful model requests, its throttle
was raised to eight. Preparation 1505945 validates exact task/seed coverage
and builds complete-history KL masks. Pilot 1505946 runs two updates and
training 1505947 resumes that checkpoint for the remaining 1,686 updates.
The 72-hour training allocation reserves 256 GiB host memory for the larger
corpus; the objective and effective batch size are unchanged.

Collection audit: shard 184 (seed 8485), task `6104387_1`, contains a
model-caused failure. The model replaced `requester.request` with an incompatible
`debug_request` function, causing the subsequent task-completion check to raise
a missing-argument error. Preserve its original saved history and error metadata;
do not reroll or omit the attempt. Assembly allows only this identified case and
checks its error signature and saved monkey-patch code. Missing task scores count
as unsuccessful in collection metrics. All export errors and other sample errors
remain fatal validation failures. The assembled manifest records the exception.

Export 1505948, fresh probe fit 1505949, evaluation collection 1505950
(indices 0–2 only), assembly 1505951 and role scoring 1505952 (indices 0–1
only) depend on successful upstream stages. These jobs do not run SWE.
Report job 1505959 follows role scoring. It validates every saved prediction,
checks all 1,688 training updates and final adapter provenance, and updates only
the design-iteration block plus the new model report and appendix. All stages completed successfully. All saved prediction arrays and cohort coverage
were independently revalidated before publication.

```bash
sbatch src/experiments/training_against_role_probes/data_iteration/collect.sbatch
sbatch --dependency=afterok:COLLECTION src/experiments/training_against_role_probes/data_iteration/prepare.sbatch COLLECTION
sbatch --time=02:00:00 --dependency=afterok:PREPARE src/experiments/training_against_role_probes/data_iteration/train.sbatch src/experiments/training_against_role_probes/data_iteration/pilot_config.json
sbatch --dependency=afterok:PILOT src/experiments/training_against_role_probes/data_iteration/train.sbatch src/experiments/training_against_role_probes/data_iteration/train_config.json --resume experiment_reports/training_against_role_probes/data_iteration/same_batch_27000/runs/training/epoch1/checkpoints/step00002
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/data_iteration/submit_evaluation.py --training TRAINING
```

## Artifacts and provenance

Training completed all 1,688 updates in 60 h 33 min on two B200 GPUs using
PyTorch 2.9.1+cu128. The cluster-local `runs/training/epoch1/manifest.json` records source
and input hashes and the original local revision `945898f87d1b65d76a456b5fd8fc4e36c4cde974`;
the equivalent reproduction code is included with this report. The base model
revision is `6cee5e81ee83917806bbde320786a8fb61efebee`. Upstream pins are
`ec333c40fd43fe991e1ebf66765051b6d7e35784` (role confusion) and
`9c3f80cc6d6d6fa2a434ca49ffdd150c4223c427` (training against probes).

[Aggregate metrics](publication_metrics.json) and [coverage checks](validation.json)
accompany this report. Export provenance (`data/post_training/model/export.json`),
training provenance, probe-fit metadata and fresh layer-12 weights
(`post_training/probes/`) remain in the cluster experiment directory below.
Probe-fit convergence warnings remain recorded; no convergence claim is made.

Large artifacts remain under
`/data/cluster/users/kortukov/role_probe_training/experiment_reports/training_against_role_probes/data_iteration/same_batch_27000/`.
Training trajectories and masks are in `data/train27000/` and `data/anchors.jsonl`;
final restart state is `runs/training/epoch1/checkpoints/step01688/`.
Evaluation trajectories are `data/post_training/{appworld_test,tau2_retail}/`;
whole-history predictions are `predictions/post_training/`. These cluster-local
artifacts require cluster access; they are not included in Git. Reproduction
commands above regenerate them from the recorded seeds and inputs.
