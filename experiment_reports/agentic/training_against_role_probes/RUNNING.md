# Reproduction

## Collection

The collection job reuses the validated AppWorld agent, environment, evaluator,
server and normalized JSONL export. Each shard has its own inference GPU and eight
concurrent worlds. Seeds 7301–7303 collect all 90 training tasks; seed 7304 collects
30 after deterministic shuffling. Sampling temperature is one. Existing 90-task
benchmark trajectories are not silently included in this new corpus.

```bash
sbatch --exclude=vca-gpu-0701-03 src/experiments/training_against_role_probes/collect.sbatch
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble.py \
  --runs experiment_reports/agentic/training_against_role_probes/runs/JOB_ID \
  --output experiment_reports/agentic/training_against_role_probes/data
```

The assembler requires complete error-free exports with 300 distinct attempt IDs
and 90 distinct tasks. Execution-limited and unsuccessful attempts remain included.
An attempt's unique ID includes its original task ID and generation seed. Generation
seeds and task sampling are recorded; concurrent GPU execution is not promised to
be bitwise deterministic. JSONL retains messages, channels, tools and scores.

Large data, logs, checkpoints and W&B local files are ignored. Keep the resulting
small corpus manifest and training metrics in Git. Collection uses the existing
locked `agentic_eval/swe` and `agentic_eval/server_current` environments; see the
[AppWorld reproduction guide](../appworld/RUNNING.md) for exact runtime details.

## Upstream reference and training status

Training source: `submodules/training_against_probes`, revision
`9c3f80cc6d6d6fa2a434ca49ffdd150c4223c427`. Relevant code:
`configs_runs/honesty_example.yaml`, `src/finetune.py` (model/refit loop),
`src/utils/finetune_utils.py` (KL), `src/probes/build_kl_rollouts.py` (anchor).

The role training loop is not implemented or launched yet: [DESIGN.md](DESIGN.md)
records the concrete adaptations awaiting discussion. W&B project intended:
`training_against_role_probes`; verify authenticated account before the first run.
No credentials or task text should be put in tracked config or W&B logs.

## Current submission

Collection array **1492660** has four one-GPU shards. CPU assembly job **1492664**
requires all four to succeed and writes the canonical JSONL plus `corpus.json`.
`submission.json` records seeds and the collection-code revision. SWE inference
1492632 now also waits for this array to finish, keeping total inference allocation
within four GPUs while SWE CPU image preparation proceeds concurrently.

W&B authentication was checked through the configured `~/.env` environment; no
`WANDB_API_KEY` was present. Online logging is not verified. A GitHub-linked W&B
account still needs runtime credentials (for example `wandb login` in the training
environment); do not paste secrets into reports or chat.
