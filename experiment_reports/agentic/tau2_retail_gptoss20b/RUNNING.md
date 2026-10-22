# Reproduction

Run from the repository root on cluster storage. Training's root environment is
untouched. Code: [runner](../../../src/experiments/agentic_eval/run.py),
[Slurm job](../../../src/experiments/agentic_eval/job.sbatch),
[exporter](../../../src/role_probe_training/trajectories/inspect_export.py).

```bash
export PATH="$PWD/.tools/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
uv sync --project src/experiments/agentic_eval --locked
uv sync --project src/experiments/agentic_eval/server --locked
uv pip install --python src/experiments/agentic_eval/.venv/bin/python --no-deps -e .
uv run --project src/experiments/agentic_eval --frozen --no-sync python src/experiments/agentic_eval/checks.py
mkdir -p experiment_reports/agentic/tau2_retail_gptoss20b/logs
sbatch --partition=ber-testing --time=00:20:00 src/experiments/agentic_eval/job.sbatch --smoke-only
sbatch --exclude=vca-gpu-0701-03 src/experiments/agentic_eval/job.sbatch --samples 20 --user-model openai/azure/gpt-4.1 --max-tokens 16384 --token-limit 500000 --time-limit 900
```

The submitting shell must export `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_ENDPOINT` from the user's `~/.bashrc`. Do not put credentials in
scripts or reports. Local model files are reused from `.cache/models/gpt-oss-20b`
(the pinned snapshot staged by the probe-training experiment).

Results live at `runs/<SlurmJobID>/`: `manifest.json`, `logs/*.eval`,
`trajectories.jsonl`, `generations.jsonl`, `summary.json`, `server.log`, `smoke.json`.
These raw artifacts are ignored in Git, retained on cluster storage; copy small
summary/manifest files into `metrics/` for the versioned report. Archive raw logs
before scratch cleanup. Each run has exact IDs, package versions, source/lock/data
hashes, seed, settings and source Git revision (working tree may be dirty).
The server uses the separate `server/uv.lock` (Torch 2.8.0, Transformers 4.57.1).
No upstream submodule is used for benchmark generation; probe revisions belong
to the later probe-evaluation report.

To re-export retained logs (including completed samples from an interrupted run):

```bash
uv run --project src/experiments/agentic_eval --frozen --no-sync python src/experiments/agentic_eval/export.py experiment_reports/agentic/tau2_retail_gptoss20b/runs/JOB_ID
```

Optional array sharding: pass `--shard-count N --shard-index I`; `--samples 20`
means 20 total across shards. Keep separate run directories and merge only
nonduplicated sample/epoch logs. Start with one GPU; more jobs repeat model load.
See [trace contract and caveats](../../../agent_docs/AGENTIC_EVALS.md).

The node exclusion reflects a live CUDA error 802 on 2026-10-21, not a permanent
cluster rule; remove it after the node is repaired. Original budget-limited run
1448719 used `--max-tokens 4096 --token-limit 100000 --time-limit 600`. Keep both
runs separate; do not pool repeated trials as independent new tasks.

Validate saved chats on CPU without loading model weights:

```bash
uv run --project src/experiments/agentic_eval/server --frozen --no-sync python src/experiments/agentic_eval/tokenize_check.py experiment_reports/agentic/tau2_retail_gptoss20b/runs/JOB_ID/trajectories.jsonl --output experiment_reports/agentic/tau2_retail_gptoss20b/metrics/tokenization.json
```

The input also supports `.jsonl.gz`. Token counts use the cached HF template;
this check does not assert equivalence to vLLM's native Harmony serving prompts.

## Saved pilot results

Main run: Slurm 1448731 on `vca-gpu-0701-02`, completed 2026-10-21; 27m01s
allocation, 25m25s evaluation. The small summary, manifest, per-task CSV and
summary artifacts are versioned in `metrics/`; compressed normalized messages remain local in `data/`.
The full raw run is retained at:
`/data/cluster/users/kortukov/role_probe_training/experiment_reports/agentic/tau2_retail_gptoss20b/runs/1448731/`.
The initial budget-limited run is the adjacent `1448719/` directory.

Read the portable archive directly:

```python
import gzip, json
with gzip.open("experiment_reports/agentic/tau2_retail_gptoss20b/data/trajectories.jsonl.gz", "rt") as f:
    trajectories = [json.loads(line) for line in f]
messages = trajectories[0]["messages"]
tools = trajectories[0]["tools"]
```

The archive is exactly the main run's `trajectories.jsonl` compressed with gzip
(mtime=0); raw and compressed SHA-256 hashes are in `metrics/trace_audit.json`.
`metrics/trajectories.csv` is a direct projection of each row's ID, score, duration,
message/tool/reasoning counts, caps, errors and limit types. `metrics/summary.json`
and `metrics/manifest.json` are copies of the run's corresponding files.
Keep the original `.eval` and `generations.jsonl` for provider-level reconstruction;
the portable archive intentionally contains only the evaluated agent's history.

## Recorded pilot configuration

- Inspect AI 0.3.282 / Inspect Evals 0.25.0, `tau2_retail`, base split (114 tasks).
  Twenty fixed tasks, seed 123, one trial per task per run. The same subset was
  used for both budget settings. [Final manifest](metrics/manifest.json).
- GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`, native MXFP4,
  vLLM 0.11.0 / Transformers 4.57.1 / Torch 2.8.0, one B200, eight concurrent
  samples. High reasoning, temperature 0, generation seed 123. Native Harmony
  serving with the OpenAI tool parser; eager execution and prefix caching.
- Azure customer simulator: deployment `gpt-4.1`, verified response model
  `gpt-4.1-2025-04-14`, temperature 0, 2,048 output tokens/call. Chat completions,
  API version 2024-10-21. Retail's scorer is deterministic: database state plus
  required communicated information. There is no model-based grading charge.
- Main run: 32,768 context tokens, 16,384 output tokens/call, 500,000 cumulative
  API tokens/sample, 900 seconds/sample, upstream 100 simulator-turn limit;
  one-hour Slurm job. The earlier pilot had tighter limits, listed below.

## Artifact index



- [Reusable trajectories.jsonl.gz](data/trajectories.jsonl.gz) — 20 full message
  records including reasoning, tools, scores, provenance and quality flags; 326 KiB.
- [Main summary](metrics/summary.json), [per-task CSV](metrics/trajectories.csv),
  [trace audit](metrics/trace_audit.json), [tokenization checks](metrics/tokenization.json).
- [Initial pilot summary](metrics/pilot_4096_summary.json) and
  [job history](metrics/jobs.json), including the corrected tool-parser smoke test
  and one production-node CUDA startup failure before inference.
- Raw `.eval` logs, per-generation JSONL and server logs remain under
  `runs/1448731/` on cluster storage; exact paths and hashes are in the manifests.
