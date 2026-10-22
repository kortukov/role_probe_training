# Reproduction

Generation runs and immutable inputs are recorded in [metrics/provenance.json](metrics/provenance.json). Their raw `.eval` logs and `generations.jsonl` retain individual model calls. The consolidated chat data is `data/trajectories.jsonl`; its hash is in [metrics/summary.json](metrics/summary.json). Data is kept on cluster storage and excluded from Git.

Combine completed disjoint shards:

```bash
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/agentic_eval/finalize.py --benchmark swe_bench --run experiment_reports/agentic/swe_bench_full/runs/1461197 experiment_reports/agentic/swe_bench_full/runs/1461195 experiment_reports/agentic/swe_bench_full/runs/1461196 experiment_reports/agentic/swe_bench_full/runs/1461193 --replacement-run experiment_reports/agentic/swe_bench_full/runs/normalized-1461193 experiment_reports/agentic/swe_bench_full/runs/normalized-1461196 experiment_reports/agentic/swe_bench_full/runs/1463740 --selection-manifest experiment_reports/agentic/swe_bench_full/runs/1461197/manifest.json --output experiment_reports/agentic/swe_bench_full
```

## Reproducing the reported full run

The isolated client and server projects are `src/experiments/agentic_eval/swe` and `src/experiments/agentic_eval/server`. Sync their locks, then install the root package into the client with `uv pip install --python src/experiments/agentic_eval/swe/.venv/bin/python --no-deps -e .`. Keep the original recorded environments for exact reproduction.

```bash
mkdir -p experiment_reports/agentic/swe_bench_full/logs
SWE_SERVER_PROJECT=server SWE_SHARD_COUNT=4 sbatch --array=0-3 --exclude=vca-gpu-0701-03 src/experiments/agentic_eval/swe_job.sbatch --all --temperature 0 --message-limit 30 --token-limit 500000
```

The two image-layer repairs are recorded in [image_layer_repairs.json](image_layer_repairs.json); `repair_image_layers.py` verifies the official blob hashes before populating the cache. The separate `swe_repair_job.sbatch` explicitly selects only those two tasks. Normalization repairs use `repair_exports.py`, preserving original logs. Finalization rejects replacements unless they repair export errors or image failures with zero prior model calls. See selected-run provenance above; never silently replace failed model attempts with successful retries.

## Probe predictions and standalone viewer

See [probe reproduction](../probe_transfer/RUNNING.md) and [layer-comparison reproduction](../probe_layer_comparison/RUNNING.md). The shared `agentic_probes/report.py` script accepts repeated `--dataset NAME=PREDICTION_DIRECTORY`, optional benchmark summary paths, and layer choices. It embeds all data in a standalone `viewer.html`, which stays local and can be downloaded and opened without a server.

## Recorded execution settings

GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`, high reasoning, temperature 0, seed 123, vLLM 0.11.0. One selected model attempt per task; all dataset tasks included. Infrastructure failures before model execution may be retried, with both attempts retained in provenance. Per-call output limit 16,384; per-sample cumulative token limit 500,000 and wall-time limit 1,800 seconds. Source manifests record exact versions, task IDs, code hashes, Slurm jobs and model configuration.

All 500 SWE-bench Verified tasks, pinned dataset revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Stock Inspect solver/scorer, 30-message cap; four shards with eight concurrent samples each. Validated Apptainer isolation and gold-patch scoring. Official images are pinned by digest in [image_manifest.json](image_manifest.json) and retrieved from Google’s public Docker Hub cache.

## Runtime and interpretation of configuration

The four successful generation shards took 72–84 minutes concurrently on four B200 GPUs. Infrastructure-only retries took another 10 minutes. Generation used temperature 0 and a 32K serving context with the stock 30-message Inspect cap. The official model repository recommends temperature 1/top-p 1; these settings were not changed midway through collection. See [model sampling guidance](https://github.com/openai/gpt-oss). This setup differs from the published terminal harness and evaluation subset.

## Serving repair check

`src/experiments/agentic_eval/swe_replay.py --base-url URL --output FILE` replays four deterministic failed contexts at temperatures 0 and 1 without executing any tools. It resolves all Inspect attachments before sending requests. The passing check used vLLM 0.32.0 from `server_current`, with the same checkpoint, Triton kernels and compiled execution configured by the AppWorld job. Source selection and results are in [metrics/serving_diagnostic.json](metrics/serving_diagnostic.json). This check must not be combined with the full-run success metric or substituted into probe inputs.

`swe_run.py` now exposes `--temperature` and `--message-limit` for a future validated rerun; defaults preserve the recorded configuration.

## Repaired end-to-end configuration

The normal `swe_job.sbatch` now defaults to `server_current`, compiled Triton inference, temperature 1 and a 200-message allowance. `SWE_SERVER_PROJECT=server` selects the original environment only for reproducing the reported 500-task run above.

Prepare selected official images on a CPU allocation before requesting GPUs:

```bash
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/agentic_eval/cache_swe_image.py astropy__astropy-14182
AGENTIC_REPORT_ROOT=experiment_reports/agentic/swe_bench_full/diagnostic sbatch --time=00:45:00 --exclude=vca-gpu-0701-03 src/experiments/agentic_eval/swe_job.sbatch --sample-id astropy__astropy-14182 --samples 1 --concurrency 1 --time-limit 1200 --max-tokens 8192
```

Skip preparation if that image is already cached. The cache contains a SIF and its source-image/hash sidecar; the sandbox checks both before use. This avoids spending GPU time on registry access and image compression. The completed check is `diagnostic/runs/1489368`; its original Inspect log, exported trace and manifest remain local. It is kept separate from the 500 selected probe-evaluation histories.
