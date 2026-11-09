# Current handoff — 2026-11-04

## Published experiments

- The [training comparison](../experiment_reports/training_against_role_probes/README.md)
  contains all seven fixed-data models, three on-policy variants and validated
  task/whole-history role results. The [seven-model SWE cohort](../experiment_reports/training_against_role_probes/swe_generous/README.md)
  is complete. Keep newer validated seven-model measurements; do not replace them
  with the earlier native-context, corrected-environment or 30-message cohorts.
- [AgentDojo robustness](../experiment_reports/prompt_injection/agentdojo_same_batch_27000/README.md)
  is complete: 95 clean tasks and 921 attacked pairs per model. Base/trained
  recorded attack success: 27.69%/29.53%. Five/eight context-limit errors have no
  final verdict; all attempts remain in denominators. No measured robustness gain.
- The complete adapter is shared at
  https://huggingface.co/kortukov/gpt-oss-20b-role-perception-appworld-27000.
  The current Hub revision contains adapter weights and helpers only. Its vLLM
  launcher reconstructs the evaluation model locally; all 603 tensors match.

## Training and implementation

- Current shared trainer: `src/role_probe_training/on_policy_training/train.py`;
  recovered collection/runtime: `on_policy_recovery/`; concurrent collection is
  configured with `collection_concurrency`. Each episode uses a fresh process
  because Inspect caches base URLs per process. Episode seed/route validation
  must remain enabled. Recovery preserves whole episodes and original failures.
- 2026-11-09: the ten uv environments were merged into three
  ([ENVIRONMENTS.md](ENVIRONMENTS.md)). On DE.BER1 run `uv sync --locked --extra vllm` and
  the probe-fitting sync before submitting anything; training now runs on torch
  2.13.0+cu130, peft 0.18.1 and Transformers 5.9.0, and has not been re-run at scale there.
- The old trainer variants were removed at d96fe7a; historical training code is
  recoverable from Git. Do not remove modules still imported by queued jobs or
  referenced by checkpoint source hashes.
- Optional KL request batching is off unless `kl_batch_tokens` is set. The saved
  96-request check shows faster execution but changed BF16 losses/gradients;
  it is not numerically identical to sequential replay. Keep per-request replay
  for exact comparison. Validation code and measurements accompany the variant.
- At the 2026-11-04 publication check there were no running user GPU jobs. Jobs
  1798695/1798700/1798701 and CPU bash jobs 1963763/1963766/1963838 remained pending.
  Recheck Slurm and saved artifacts before interpreting or changing this state;
  this publication does not restart, cancel or edit any queued jobs.

## Fixed-data same-batch epoch continuation

The separate [same-batch continuation](../experiment_reports/training_against_role_probes/same_batch_epochs/README.md)
restores the original 2,700-trajectory checkpoint and optimizer and reuses the
same data through five total epochs. Its isolated trainer and frozen job sources
leave paired rollout-refresh experiments unchanged. Training job 2020991 and
epoch-2/3/5 evaluations 2020992/2020993/2020994 were submitted on 2026-11-04;
these are submission records, not verified current scheduler states. Saved
epoch-1 AppWorld and retail measurements are validated; later rows remain pending.
See the experiment RUNNING.md and jobs.json for provenance and source snapshots.

## Standing workflow

Use uv and cluster storage; details in [CLUSTER.md](CLUSTER.md). Publish small
reports/provenance and code, retain large trajectories/checkpoints on cluster
storage with recorded paths. Keep all assigned attempts in task denominators;
replay role histories whole and report over-context coverage. Follow
[REPORT_WRITING.md](REPORT_WRITING.md). Do not launch training just to validate
publication. Superseded SWE investigations are stopped; do not rearm their watchers.
