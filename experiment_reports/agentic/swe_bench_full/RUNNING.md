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

## Recorded execution settings

GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`, high reasoning, temperature 0, seed 123, vLLM 0.11.0. One selected model attempt per task; all dataset tasks included. Infrastructure failures before model execution may be retried, with both attempts retained in provenance. Per-call output limit 16,384; per-sample cumulative token limit 500,000 and wall-time limit 1,800 seconds. Source manifests record exact versions, task IDs, code hashes, Slurm jobs and model configuration.

All 500 SWE-bench Verified tasks, pinned dataset revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Stock Inspect solver/scorer, 30-message cap; four shards with eight concurrent samples each. Validated Apptainer isolation and gold-patch scoring. Official images are pinned by digest in [image_manifest.json](image_manifest.json) and retrieved from Google’s public Docker Hub cache. The earlier rate-limited attempt is preserved but excluded from these results.

## Runtime and interpretation of configuration

The four successful generation shards took 72–84 minutes concurrently on four B200 GPUs. Infrastructure-only retries took another 10 minutes. Generation used temperature 0 and a 32K serving context with the stock 30-message Inspect cap. The official model repository recommends temperature 1/top-p 1; these settings were not changed midway through collection. See [model sampling guidance](https://github.com/openai/gpt-oss). This setup differs from the published terminal harness and evaluation subset.
