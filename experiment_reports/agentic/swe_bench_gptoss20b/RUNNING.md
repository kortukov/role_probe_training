# Preparation and eventual execution

Use the client/server environment setup in [retail RUNNING](../tau2_retail_gptoss20b/RUNNING.md).
The shared [runner](../../../src/experiments/agentic_eval/run.py) accepts:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
uv run --project src/experiments/agentic_eval --frozen --no-sync python src/experiments/agentic_eval/run.py --benchmark swe_bench --samples 5 --prepare-only --output experiment_reports/agentic/swe_bench_gptoss20b/preparation
```

On a validated Docker-capable allocated execution host, with a reachable GPT-OSS
server, remove `--prepare-only`, choose a **fresh** output directory, and add
`--concurrency 2 --time-limit 900 --base-url http://SERVER:PORT/v1`.
Do not expose an unauthenticated server publicly. The current retail job binds
localhost; it needs no network exposure. No SWE-bench cluster job is submitted.

The exact selected task IDs, dataset checksum and source versions are recorded
in [preparation/manifest.json](preparation/manifest.json). Inspect creates compose
specifications in `.cache/inspect-evals/swe_bench/compose_files`. Pull only selected
images and record immutable image digests before evaluation. Keep per-instance
patches/tests and original `.eval` logs as well as the common JSONL exports.

[Operational notes](../../../agent_docs/AGENTIC_EVALS.md)
