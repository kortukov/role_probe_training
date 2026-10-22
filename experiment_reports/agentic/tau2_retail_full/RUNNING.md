# Reproduction

Generation runs and immutable inputs are recorded in [metrics/provenance.json](metrics/provenance.json). Their raw `.eval` logs and `generations.jsonl` retain individual model calls. The consolidated chat data is `data/trajectories.jsonl`; its hash is in [metrics/summary.json](metrics/summary.json). Data is kept on cluster storage and excluded from Git.

Combine completed disjoint shards:

```bash
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/agentic_eval/finalize.py --benchmark tau2_retail --run experiment_reports/agentic/tau2_retail_full/runs/1459495 --selection-manifest experiment_reports/agentic/tau2_retail_full/runs/1459495/manifest.json --output experiment_reports/agentic/tau2_retail_full
```

## Launching a full run

Run from the repository root. The submitting shell must export `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`; credentials are inherited by Slurm and never saved in configuration or reports.

```bash
export PATH="$PWD/.tools/bin:$PATH"
uv sync --project src/experiments/agentic_eval --locked
uv sync --project src/experiments/agentic_eval/server --locked
uv pip install --python src/experiments/agentic_eval/.venv/bin/python --no-deps -e .
mkdir -p experiment_reports/agentic/tau2_retail_full/logs
sbatch src/experiments/agentic_eval/job.sbatch --all --concurrency 16 --time-limit 1800 --user-model openai/azure/gpt-4.1
```

The pinned GPT-OSS model is staged at `.cache/models/gpt-oss-20b`. The job creates a fresh `runs/<job-id>/` directory containing source Inspect logs, exported conversations, per-call records, scores and configuration. Use the corresponding manifest when consolidating a new run.

## Probe predictions and standalone viewer

See [probe reproduction](../probe_transfer/RUNNING.md) and [layer-comparison reproduction](../probe_layer_comparison/RUNNING.md). The shared `agentic_probes/report.py` script accepts repeated `--dataset NAME=PREDICTION_DIRECTORY`, optional benchmark summary paths, and layer choices. It embeds all data in a standalone `viewer.html`, which stays local and can be downloaded and opened without a server.

## Recorded execution settings

GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`, high reasoning, temperature 0, seed 123, vLLM 0.11.0. One attempt per task in this selected run; all dataset tasks included. Per-call output limit 16,384; per-sample cumulative token limit 500,000 and wall-time limit 1,800 seconds. Source manifests record exact versions, task IDs, code hashes, Slurm jobs and model configuration.

114 τ²-bench retail tasks; Azure GPT-4.1 customer simulator (`gpt-4.1-2025-04-14`), 16 concurrent samples. The original upstream 100-turn cap is retained.

## Runtime and interpretation of configuration

The full generation job took 85 minutes on one B200; scoring all five probe layers took 25 minutes together. Per-call and cumulative budgets differ: cumulative API usage includes repeated prompts and simulator calls.
