# Reproduction

Config: `src/experiments/agentic_probes/train_layers.json`. Input documents copied byte-for-byte from the baseline `artifacts/inputs/documents.jsonl`; prepared config changes only the layer selection.

```bash
sbatch src/experiments/gptoss_role_probes/train.sbatch --stage train --config src/experiments/agentic_probes/train_layers.json --output experiment_reports/role_probes/gptoss20b_layers
```
