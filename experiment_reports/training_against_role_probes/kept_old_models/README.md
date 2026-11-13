# Old-template models kept for now

Final training checkpoints (LoRA adapter, optimizer state and training probe) of the three
best models trained before the unified setup. All other run data of the old-template
experiments was deleted on 2026-11-13; these are kept in case a comparison is needed and
are to be deleted later. The checkpoint files are on cluster storage only, not in Git.

| Folder | Model | Results |
|---|---|---|
| `off_policy_same_batch_27000_step01688/` | off-policy, same-batch KL, 27,000 trajectories | [report](../data_iteration/same_batch_27000/README.md) |
| `off_policy_same_batch_2700_step00169/` | off-policy, same-batch KL, 2,700 trajectories, one epoch | [report](../kl_iteration/same_batch/README.md) |
| `on_policy_270_batch16_step00017/` | on-policy, 270 episodes, batch 16 | [report](../on_policy/appworld270_batch16/README.md) |

They were trained with the old chat templates and a layer-12 probe refitted by SGD; see
[the old results](../OLD_TEMPLATE_RESULTS.md).
