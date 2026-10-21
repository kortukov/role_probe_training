# Reproduction

Generation runs and immutable inputs are recorded in [metrics/provenance.json](metrics/provenance.json). Their raw `.eval` logs and `generations.jsonl` retain individual model calls. The consolidated chat data is `data/trajectories.jsonl`; its hash is in [metrics/summary.json](metrics/summary.json). Data is kept on cluster storage and excluded from Git.

Combine completed disjoint shards:

```bash
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/agentic_eval/finalize.py --benchmark swe_bench --run experiment_reports/agentic/swe_bench_full_mirror/runs/1461197 experiment_reports/agentic/swe_bench_full_mirror/runs/1461195 experiment_reports/agentic/swe_bench_full_mirror/runs/1461196 experiment_reports/agentic/swe_bench_full_mirror/runs/1461193 --replacement-run experiment_reports/agentic/swe_bench_normalization/runs/1461193 experiment_reports/agentic/swe_bench_normalization/runs/1461196 experiment_reports/agentic/swe_bench_image_retry/runs/1463740 --selection-manifest experiment_reports/agentic/swe_bench_full_mirror/runs/1461197/manifest.json --output experiment_reports/agentic/swe_bench_full
```

## Launching a fresh full run

The isolated client and server projects are `src/experiments/agentic_eval/swe` and `src/experiments/agentic_eval/server`. Sync their locks, then install the root package into the client with `uv pip install --python src/experiments/agentic_eval/swe/.venv/bin/python --no-deps -e .`. Keep the original recorded environments for exact reproduction.

```bash
mkdir -p experiment_reports/agentic/swe_bench_full_mirror/logs
sbatch --array=0-3 --exclude=vca-gpu-0701-03 src/experiments/agentic_eval/swe_mirror_job.sbatch
```

The two image-layer repairs are recorded in [image_layer_repairs.json](image_layer_repairs.json); `repair_image_layers.py` verifies the official blob hashes before populating the cache. The separate `swe_repair_job.sbatch` explicitly selects only those two tasks. Normalization repairs use `repair_exports.py`, preserving original logs. Finalization rejects replacements unless they repair export errors or image failures with zero prior model calls. See selected-run provenance above; never silently replace failed model attempts with successful retries.

## Probe predictions and standalone viewer

See [probe reproduction](../probe_transfer/RUNNING.md) and [layer-comparison reproduction](../probe_layer_comparison/RUNNING.md). The shared `agentic_probes/report.py` script accepts repeated `--dataset NAME=PREDICTION_DIRECTORY`, optional benchmark summary paths, and layer choices. It embeds all data in a standalone `viewer.html`, which stays local and can be downloaded and opened without a server.
