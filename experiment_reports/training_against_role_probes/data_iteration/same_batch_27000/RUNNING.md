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
retail will use fresh model-generated histories and fresh standard-data probes;
SWE remains deferred to the separate evaluation effort.

## Submitted pipeline

Job IDs are recorded in [jobs.json](jobs.json). Collection array 1505943
uses indices 30–299 and seeds 8331–8600. It started with a single active
shard; after worker initialization and successful model requests, its throttle
was raised to eight. Preparation 1505945 validates exact task/seed coverage
and builds complete-history KL masks. Pilot 1505946 runs two updates and
training 1505947 resumes that checkpoint for the remaining 1,686 updates.
The 72-hour training allocation reserves 256 GiB host memory for the larger
corpus; the objective and effective batch size are unchanged.

Export 1505948, fresh probe fit 1505949, evaluation collection 1505950
(indices 0–2 only), assembly 1505951 and role scoring 1505952 (indices 0–1
only) depend on successful upstream stages. These jobs do not run SWE.
Report job 1505959 follows role scoring. It validates every saved prediction,
checks all 1,688 training updates and final adapter provenance, and updates only
the design-iteration block plus the new model report and appendix. Publication
requires reviewing these validated results and preserving concurrent SWE edits.

```bash
sbatch src/experiments/training_against_role_probes/data_iteration/collect.sbatch
sbatch --dependency=afterok:COLLECTION src/experiments/training_against_role_probes/data_iteration/prepare.sbatch COLLECTION
sbatch --time=02:00:00 --dependency=afterok:PREPARE src/experiments/training_against_role_probes/data_iteration/train.sbatch src/experiments/training_against_role_probes/data_iteration/pilot_config.json
sbatch --dependency=afterok:PILOT src/experiments/training_against_role_probes/data_iteration/train.sbatch src/experiments/training_against_role_probes/data_iteration/train_config.json --resume experiment_reports/training_against_role_probes/data_iteration/same_batch_27000/runs/training/epoch1/checkpoints/step00002
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/data_iteration/submit_evaluation.py --training TRAINING
```
