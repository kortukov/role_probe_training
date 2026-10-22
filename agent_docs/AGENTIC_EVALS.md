# Agentic evaluation workflow

## Current experiments

`experiment_reports/agentic/` contains four current experiments:

- `tau2_retail_full`: 114 tasks; 73 solved (64.04%).
- `swe_bench_full`: 500 tasks; 11 solved (2.20% counting errors as failures).
- `probe_transfer`: layer-12 role recognition on those 614 histories.
- `probe_layer_comparison`: layers 4, 8, 12, 16 and 20 on the same histories.

SWE has 226 model API retry failures and substantial execution-limit effects.
Its score is not a matched reproduction of the model's published capability.
Keep these limitations visible in the current report. Scientific writing and
stale-material removal follow [REPORT_WRITING.md](REPORT_WRITING.md).

## Environments and execution

Use Inspect for execution, logs, scores and concurrency. Clients and the vLLM
server use separate locked uv projects because their dependency requirements
differ. Retail uses `src/experiments/agentic_eval`; SWE uses its `swe/` project
with swebench 4.1.0; serving uses its `server/` project with vLLM 0.11.0. After
syncing a client lock, install the root package into that client with
`uv pip install --python CLIENT/.venv/bin/python --no-deps -e .`, then use
`uv run --frozen --no-sync`. Do not modify the probe-training environment.

`job.sbatch` launches retail and `swe_job.sbatch` launches SWE. All output lives
under the current benchmark's `runs/` and `logs/` directories. The runners support
fixed selection and disjoint sharding. Use fresh run directories and reject
duplicate task IDs when consolidating. Generation batches across conversations;
each conversation remains sequential.

Retail uses Azure `openai/azure/gpt-4.1` as customer simulator, verified as
`gpt-4.1-2025-04-14`, chat completions API version 2024-10-21. Credentials exported
from the user's shell are inherited by Slurm and never written to manifests.
Retail scoring checks database state and required communication; it needs no
separate model grader.

SWE uses a validated Apptainer provider with writable per-task environments,
isolated networking, no host-workspace mount and no API credentials inside.
Gold-patch scoring and Inspect Python/bash-session/editor tools passed validation.
Use official task images pinned by OCI digest in `swe_bench_full/image_manifest.json`;
`apptainer_mirror.py` retrieves them through mirror.gcr.io. This cache is a runtime
source, not a separate experiment. Verified missing blobs can be seeded into the
Apptainer cache with `repair_image_layers.py`. Retry infrastructure failures only
when no model call occurred, never because an attempted solution failed.

Generation took 85 minutes on one B200 for retail and 72–84 minutes across four
concurrent B200 shards for SWE. Five-layer scoring took 25 minutes for retail and
about 15 minutes across four SWE shards. Exact limits and settings are in RUNNING
files and configurations, not the scientific narratives.

## Trace contract

`trajectories.jsonl` has one row per benchmark/sample/epoch with `messages`,
`tools`, scores, errors, limits and source provenance. Messages contain a role
and string content; assistant `thinking` retains reasoning. Tool calls use
OpenAI/HF-style function dictionaries; replies retain names and call IDs.
Missing tool names use `__unknown_tool__`, `name_missing: true`, and
`original_function`, preserving the history instead of dropping it. Never infer
roles or channels from the text's meaning.

For τ², export `sample.store['StoredAgentState:RetailAgent:messages']`.
`sample.messages` belongs to the privileged customer simulator and must not be
used as the agent history. `generations.jsonl` preserves individual evaluated-model
calls; original `.eval` logs retain full scoring/event context.

Count nested limit events and per-call output caps, not only `sample.limit`.
A completed log does not guarantee an uninterrupted conversation. Report all
selected tasks and distinguish scored-only rates from errors-as-failures rates.
`finalize.py` consolidates current records with explicit guarded replacements and
writes data/metrics while preserving authored reports.

## Probe replay and artifacts

`trajectories/replay.py` renders standardized retrospective Harmony, preserving
reasoning and tool metadata. This is not exact generation-time context recovery.
Tool schemas are synthetic system context excluded from accuracy. Original
message content receives labels from roles and channels; structural tokens are
excluded. Native serving templates can omit earlier reasoning, so retain source
logs for any future exact-context analysis.

Capture zero-based post-attention-layernorm outputs, the pre-MLP inputs. The
model is dequantized to BF16 with original eager attention. All five probes read
one forward pass per 16,384-token segment. Segments retain every token but reset
position/context. A full segment peaked at 136.67 GiB on B200. Preserve the
probe's model revision, activation site and class order.

Canonical per-token predictions are under `probe_transfer/predictions/retail/`
and `predictions/swe/`. SWE selection metadata validates messages, tools, scores,
errors and source logs against the current canonical dataset. Raw artifacts
needed by the selected result belong under `swe_bench_full/runs/`; superseded
experiments and excluded predictions do not need separate folders.

`agentic_probes/report.py` refreshes metrics, appendix and offline viewer, leaving
README/RUNNING prose intact. Compressed trajectories occupy separate inert HTML
blocks to avoid JavaScript source-size limits; decode only the selected history.
`viewer_check.py` validates all datasets/layers and the longest histories offline.
Use isolated Playwright so production locks remain unchanged. Keep data-filled
HTML, raw histories and predictions out of Git; version code, current aggregate
results and compact provenance.

## AppWorld

The [AppWorld experiment](../experiment_reports/agentic/appworld/README.md) collects
training and both test splits separately. Each world needs its own process;
threads or async worlds in one process are unsafe because AppWorld changes global
time and database state. Our Inspect adapter uses one network-isolated Apptainer
worker per world, sharing a host-side GPU inference server. Two parallel training
reference solutions passed. Keep benchmark data and derived task content out of
Git, as requested by the benchmark authors. Tune only on train/dev; inspect test
results in aggregate. Measure rollout time with the parent's monotonic clock,
not AppWorld's simulated time. Training-split collection does not train weights.
