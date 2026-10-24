# KL-data iteration

The role-training data remain the original 2,700 AppWorld train trajectories.
Each variant starts from the base model, runs one epoch (169 updates), and retains
layer 12 pre-MLP role loss, effective batch 16, LoRA rank 64/alpha 128, learning
rate 5e-5, KL coefficient 16. The fixed pool samples 16 KL examples per update; same-batch KL uses the exact role batch (16 trajectories, or 12 in the final update). All standard
probe refits retain 250 passages, their original split, all content tokens and
2,000 optimizer steps. The final evaluation fits a fresh independent probe.

`agentic800` selects 800 existing base-model train trajectories with seed 42,
without success filtering. These overlap the role-training corpus intentionally;
no test or dev trajectories enter KL. KL scores reasoning and assistant content,
including tool-call arguments. Tool responses and system/user messages are context.
`same_batch` uses all 2,700 AppWorld trajectories over the epoch, applying KL to
the exact batch used for each role-loss update. It scores the same reasoning and
assistant content mask as the fixed pool. No C4/Dolma examples enter either KL
loss; those passages are used only for the unchanged probe-refitting procedure.

Both use KL(base || adapted) with the original unshifted position convention.
Unlike the original Dolci pool, explicit masks preserve full histories without
the 2,048-token cap. Consequently this compares the KL data and its retained
context, not dataset identity alone. These are fixed base-model histories, not
new on-policy rollouts from the evolving adapter. Synthetic tool schemas are
context only and never KL targets. Metadata-labelled replay matches the role
training representation; it is not a claim of byte-identical serving prompts.

Code and configs: [iteration entry points](../../../src/experiments/training_against_role_probes/kl_iteration/).
The existing training module accepts either original prompt/completion anchors
or explicit full-history token/mask pairs. Input manifests beside each run record
source hashes, selected IDs, context lengths and scored token counts. Large data,
models and predictions remain on cluster storage under this directory, ignored
by Git. Per-run training manifests record source hashes and model revision.

Prepare the locked environment and each pool:

```bash
.tools/bin/uv sync --locked --project src/experiments/training_against_role_probes/runtime
sbatch src/experiments/training_against_role_probes/kl_iteration/prepare.sbatch agentic800
sbatch src/experiments/training_against_role_probes/kl_iteration/prepare.sbatch same_batch
```

Each two-update pilot uses the production corpus, objectives and probe-refit
budget. After it succeeds, resume `checkpoints/step00002` with the corresponding
full configuration to finish the same epoch; do not restart or reduce the data.
Training uses two B200 GPUs. Exclude `vca-gpu-0701-03,vca-gpu-0701-05`.

```bash
sbatch --exclude=vca-gpu-0701-03,vca-gpu-0701-05 src/experiments/training_against_role_probes/train.sbatch src/experiments/training_against_role_probes/kl_iteration/agentic800_pilot_config.json
sbatch --exclude=vca-gpu-0701-03,vca-gpu-0701-05 src/experiments/training_against_role_probes/train.sbatch src/experiments/training_against_role_probes/kl_iteration/agentic800_config.json --resume experiment_reports/training_against_role_probes/kl_iteration/agentic800/runs/training/epoch1/checkpoints/step00002
```

For `same_batch`, use its corresponding config paths and the isolated
`kl_iteration/train_same_batch.sbatch` wrapper. It invokes
`role_probe_training.training.train_same_batch`, a copy of the trainer whose only
objective change is selecting the exact role batch for KL. The shared fixed-pool
trainer remains unchanged while that experiment is active. Startup requires exact
trajectory-ID coverage and identical role/KL token IDs. Final checkpoints
are `VARIANT/runs/training/epoch1/checkpoints/step00169`. Evaluation will use
168 AppWorld normal-test tasks, 114 retail tasks and 500 SWE Verified tasks,
with the corrected native-context SWE protocol. Score complete histories;
retain over-native-context attempts in task denominators and report role coverage.
The existing report's base/AppWorld rows follow its canonical comparison file.
Refresh the separate iteration section without editing the four-model table:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/report.py
```

## Submitted jobs

Authoritative dependency graphs: [agentic800/jobs.json](agentic800/jobs.json) and
[same_batch/jobs.json](same_batch/jobs.json). Preparation jobs are 1504826 and 1504850.
GPU pilots are 1504827 and 1504851; full epochs are 1504830 and 1504852.
Both preparation jobs passed the CPU KL direction/gradient and full-history mask
checks. Agentic pilot 1504827 completed both updates and all probe refits; the
full epoch resumed successfully from step 2. Its [pilot validation](agentic800/pilot_validation.json)
records timing and memory, not benchmark results.
Each full run resumes the pilot's two completed updates, for 169 total updates.

Export → fresh probe fit and benchmark generation → complete cohort assembly →
whole-history scoring → validation/reporting are submitted with `afterok`
dependencies. AppWorld and retail use the existing evaluation settings. SWE uses
the corrected native 131,072-context protocol and is additionally gated on the
existing scorer/context validation job 1504807. Each collection/scoring array
runs at most two GPU shards at once. No old SWE attempts are mixed into new rows.
The reports are written only after all three cohorts pass hash/coverage checks.

To submit the same evaluation graph for a new training job in a fresh variant
folder, use `submit_evaluation.py VARIANT --training JOB --native-gate GATE` from
the iteration entry-point directory. It refuses to overwrite an existing job
manifest. Do not resubmit the graph while its recorded jobs are active.
