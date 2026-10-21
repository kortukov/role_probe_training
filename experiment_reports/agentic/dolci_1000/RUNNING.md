# Reproduction

See `src/experiments/agentic_eval/` for benchmark/download scripts and separate locked environments; `src/experiments/agentic_probes/` for inference and reports.

The shared data format is JSONL with `messages`, tool metadata, IDs, scores and source provenance. Raw outputs live under this report tree in ignored `runs/`, `data/`, or `predictions/` directories.

## Standalone viewer

Run `uv run --frozen --no-sync python src/experiments/agentic_probes/report.py --dataset NAME=PREDICTION_DIRECTORY --layers 12 --output REPORT_DIRECTORY`. Repeat `--dataset` for more inputs; use `--layers 4 8 12 16 20` for the separate layer comparison. The generated `viewer.html` embeds all data, needs no server/network, and is excluded from Git.
