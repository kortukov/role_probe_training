# Agentic evaluation workflow

## Current experiments

`experiment_reports/eval_role_probes/agentic_eval/` mirrors the code layout in
`src/experiments/eval_role_probes/agentic_eval/`, one folder per benchmark:

- `tau2_retail/base_trajectories`: 114 tasks; 73 solved (64.04%).
- `swe_bench/base_trajectories`: 500 tasks; 11 solved (2.20% counting errors as failures).
- `appworld/base_trajectories`: separate train, normal-test and challenge-test collections
  (coauthor's cluster); `appworld/base_trajectories_hx2`: the normal-test cohort regenerated on hx2;
  `appworld/probe_transfer`: probe scores on the former.
- `probe_scoring/probe_transfer`: layer-12 role recognition on the 614 retail and SWE histories.
- `probe_scoring/probe_layer_comparison`: layers 4, 8, 12, 16 and 20 on the same histories.

SWE has 226 model API retry failures and substantial execution-limit effects.
Its score is not a matched reproduction of the model's published capability.
Keep these limitations visible in the current report. Scientific writing and
stale-material removal follow [REPORT_WRITING.md](REPORT_WRITING.md).

## Environments and execution

Use Inspect for execution, logs, scores and concurrency. Inspect never talks to vLLM
directly: it talks to the Harmony proxy (`shared/harmony_proxy.py`, root environment),
which renders every prompt with oss-harmony, replays earlier assistant turns from their
generated tokens, samples token IDs from vLLM's completions endpoint and records them.
See [HARMONY_RENDERING.md](../HARMONY_RENDERING.md). Inspect (inspect-ai 0.3.283,
inspect-evals 0.25.0, swebench 4.1.0), the proxy and vLLM all live in the root
environment, installed with `uv sync --locked --extra vllm`
([ENVIRONMENTS.md](ENVIRONMENTS.md)); jobs use `uv run --frozen --no-sync` with no
`--project`. vLLM is 0.27.1 with compiled Triton kernels, the version of Xiaoxue's
`prompt_injection_interp` project (the recorded cohorts used 0.32.0 and 0.11.0 from
separate server environments). The proxy only uses vLLM's completions endpoint by token
ID. `shared/hx2/serving_smoke.sbatch` checks that path, one tool round trip through the
proxy and one Inspect request. Only the AppWorld package itself keeps its own project,
`appworld/env/`.

`job.sbatch` launches retail and `swe_job.sbatch` launches SWE. Each job starts vLLM,
then one Harmony proxy on `port + 10000` with the run's reasoning effort, `max_tokens`
and temperature (1 for every collection since 2026-11-13; `AGENTIC_TEMPERATURE=0` on the retail, AppWorld and
generous SWE-bench jobs decodes greedily; the retail user simulator stays at 0), runs
`shared/smoke.py` and the runner against the proxy with `--requests
$run_dir/proxy/requests.jsonl`, and re-exports with that file on exit. The proxy rejects
a request whose settings differ, and after any error it refuses every further request,
so one failed conversation stops the whole collection loudly rather than leaving gaps.
The SWE collection scripts under `src/experiments/training_against_role_probes/swe_*/`
still bypass the proxy and must not be used for new cohorts until changed. All output lives
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
Use official task images pinned by OCI digest in `swe_bench/base_trajectories/image_manifest.json`;
`apptainer_mirror.py` retrieves them through mirror.gcr.io. This cache is a runtime
source, not a separate experiment. Verified missing blobs can be seeded into the
Apptainer cache with `repair_image_layers.py`. For new SWE runs, prepare selected
SIFs on a CPU allocation with `cache_swe_image.py`; `SWE_IMAGE_CACHE` enables
verified local images and avoids GPU-side registry/packaging stalls. Retry infrastructure failures only
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

The probes do not read `messages`. Each row also carries `history`: the Harmony
proxy's record of the sample's last model call (`prompt_ids`, `completion_ids`,
`prompt_messages`, `completion_messages`, `rendering`), joined by conversation key
from the proxy's `requests.jsonl`. The key hashes the leading system message and the
first user message; two samples with the same key abort the export, so run epochs
separately. `history: null` means no model call happened
(context limit before the first generation); such rows stay in task-success
denominators and are unscorable for role metrics. `unlabelled_generated_tokens`
counts generated content whose channel gives no role.

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

`trajectories/replay.py::encode_record` reads a row's `history` token record: the
exact prompt and completion of the last model call, so scoring sees precisely the
tokens the model saw and wrote. Nothing is re-rendered; rows without a record are
rejected and must be regenerated through the proxy. Labels come from message
roles and channels of the recorded spans; header and control tokens are excluded.
The Harmony system message and the tool-schema block of the developer message have
`source_message_index=-1` and are excluded from accuracy.

Capture zero-based post-attention-layernorm outputs, the pre-MLP inputs. The
model is dequantized to BF16 with original eager attention. All five probes read
one forward pass per 16,384-token segment. Segments retain every token but reset
position/context. A full segment peaked at 136.67 GiB on B200. Preserve the
probe's model revision, activation site and class order.

Canonical per-token predictions are under `probe_transfer/predictions/retail/`
and `predictions/swe/`. SWE selection metadata validates messages, tools, scores,
errors and source logs against the current canonical dataset. Raw artifacts
needed by the selected result belong under `swe_bench/base_trajectories/runs/`; superseded
experiments and excluded predictions do not need separate folders.

`probe_scoring/report.py` refreshes metrics, appendix and offline viewer, leaving
README/RUNNING prose intact. Compressed trajectories occupy separate inert HTML
blocks to avoid JavaScript source-size limits; decode only the selected history.
`viewer_check.py` validates all datasets/layers and the longest histories offline.
Use isolated Playwright so production locks remain unchanged. Keep data-filled
HTML, raw histories and predictions out of Git; version code, current aggregate
results and compact provenance.

## AppWorld

The [AppWorld experiment](../experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories/README.md) collects
training and both test splits separately. Each world needs its own process;
threads or async worlds in one process are unsafe because AppWorld changes global
time and database state. Our Inspect adapter uses one network-isolated Apptainer
worker per world, sharing a host-side GPU inference server. Two parallel training
reference solutions passed. Keep benchmark data and derived task content out of
Git, as requested by the benchmark authors. Tune only on train/dev; inspect test
results in aggregate. Measure rollout time with the parent's monotonic clock,
not AppWorld's simulated time. Training-split collection does not train weights.

`appworld_run.py` takes `--base-url` of the proxy and `--requests`, the proxy's
`requests.jsonl`; `hx2/appworld.sbatch` starts one proxy per collection.
AppWorld scoring must run even when Inspect interrupts generation at a limit.
The adapter evaluates the final world state in a shielded `finally` block before
closing its worker; `appworld_check.py` verifies this and parallel oracle solutions.
The two-GPU launcher uses independent replicas and disjoint shards, not tensor
parallelism. Batch latency and steady collection throughput are different; use
the current report for measured values rather than extrapolating from GPU count.
