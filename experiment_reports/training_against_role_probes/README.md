# Training against role probes: status

Stage 2 of the project: train GPT-OSS-20B so that a role probe, refitted after every
update, recognises the role of every token, while task success stays at the base model's
level. The probe setup comes from [stage 1](../eval_role_probes/README.md).

## Where we are

All numbers in %. Role accuracy is the share of content tokens whose role a freshly fitted
probe predicts correctly; task success counts every assigned task. Order within a cell:
AppWorld test / τ² retail / SWE-bench Verified.

| Training setup | Template | Role accuracy | Task success | Status |
|---|---|---|---|---|
| Base model | old | 26.5 / 19.9 / 41.2 | 26.8 / 64.0 / 50.8 | reference, old template |
| Off-policy, same-batch KL, 2,700 trajectories | old | 99.3 / 78.8 / 93.9 | 36.9 / 54.4 / 48.6 | old template |
| Off-policy, same-batch KL, 27,000 trajectories | old | 99.8 / 77.9 / 97.7 | 29.8 / 54.4 / 51.4 | best so far, old template |
| On-policy, 270 episodes, batch 16 | old | 87.3 / 46.2 / 46.0 | 32.7 / 55.3 / 44.0 | not working yet, old template |
| Base model | new | see [stage 1](../eval_role_probes/README.md) | 21.4 / pending / 52.0 | current reference |
| **Run 1:** off-policy, same-batch KL, 2,700 trajectories | new | | | **next** |
| Run 2: the same with 27,000 trajectories | new | | | after run 1 |
| Run 3: on-policy | new | | | after run 2 |

What the old-template work established: training on fixed base-model rollouts with a KL
term computed on the same batch raises role accuracy to 78–99% and keeps SWE-bench task
success at the base level; without that KL term task success collapses; on-policy training
did not yet reach comparable role accuracy. Every old-template experiment is in
[OLD_TEMPLATE_RESULTS.md](OLD_TEMPLATE_RESULTS.md) and the folders it links.

## What to run next

Run 1 repeats the best off-policy recipe under the current setup. Only the items marked
*changed* differ from the old 2,700-trajectory run.

| Item | Setting |
|---|---|
| Model | GPT-OSS-20B, LoRA rank 64 (alpha 128) on attention and expert projections |
| Training data | 2,700 base-model rollouts: 30 attempts (seeds 8301–8330) on each of the 90 AppWorld training tasks, no filtering by outcome |
| Rollout generation, *changed* | regenerated through the Harmony proxy (the old saved rollouts have no token records and cannot be used); temperature 1, high reasoning, 8,192 output tokens, 32,768 context, 50 steps |
| Loss | role cross-entropy through the probe, plus KL(current ‖ base) on the model's own tokens of the same batch, coefficient 16 (`"kl_sampling": "same_role_batch"`) |
| Optimisation | 1 epoch, effective batch 16 (169 updates), learning rate 5e-5, seed 123, whole histories without truncation |
| Probe layer, *changed* | 16 (planned; confirm against the [stage 1 table](../eval_role_probes/README.md) once SWE-bench is scored). The old runs used 12 |
| Starting probe, *changed* | `experiment_reports/gptoss_role_probes/lbfgs_fitter/probes/layer_16.npz` |
| Probe refit, *changed* | after every update, L-BFGS to convergence on the 250 neutral passages (`lbfgs_fitter/artifacts/inputs/documents.jsonl`); the old runs took 2,000 SGD steps |
| Evaluation, *changed* | fresh converged probe per checkpoint; AppWorld test, τ² retail, SWE-bench, all at temperature 1; compare with the new base-model row |

Run 2 changes only the data: 300 attempts per task (seeds 8301–8600, 27,000 rollouts,
1,688 updates; the old run took 60 hours on two B200 GPUs). Run 3 is on-policy training
with the current trainer in `src/role_probe_training/on_policy_recovery/`.

Run 1 is prepared as its own experiment: [off_policy_2700](off_policy_2700/README.md), with
the four commands to launch it in its [RUNNING.md](off_policy_2700/RUNNING.md) (rollout
collection through the Harmony proxy, assembly and KL masks, a two-update pilot, the full
run). Code: `src/experiments/training_against_role_probes/off_policy_2700/`.

Checked without a GPU; not yet run on one. Evaluating the trained model on the three
benchmarks still needs the old evaluation pipeline (`post_training/`) adapted to the
Harmony proxy.

## Folders

| Folder | Contents | Status |
|---|---|---|
| `off_policy_2700/` | run 1: same-batch KL on 2,700 rollouts under the unified setup | current, prepared |
| `kl_iteration/`, `data_iteration/` | off-policy runs with agentic and same-batch KL; 2,700 and 27,000 trajectories | old template |
| `same_batch_epochs/` | more epochs on the fixed 2,700 trajectories | old template |
| `off_policy_epochs/`, `on_policy_epochs/` | ten-epoch comparison of the two rollout-refresh schedules | old template |
| `on_policy/` | on-policy pilots (270 and 2,700 episodes) | old template |
| `astra_sft_1k/`, `depthbench/` | training on other datasets | old template |
| `swe_generous/`, `swe_corrected/`, `swe_gap/`, `swe_standard/` | SWE-bench evaluation of the old models; `swe_generous` holds the reported numbers, the others are superseded | old template |
| `base_model/` | base-model reference scores; also holds the BF16 model export that SWE-bench serving and probe scoring still read | old template, export in use |
| `post_training/` | evaluation pipeline shared by the old models | old template |
