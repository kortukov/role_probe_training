# Agentic evaluation workflow

Verified on 2026-10-21 against inspect-evals 0.25.0 and the locked Inspect AI
version in `src/experiments/agentic_eval/uv.lock`.

## Scope and environments

Use Inspect for common execution, logs, scores and concurrency. Keep benchmark
adapters small. `src/role_probe_training/trajectories/inspect_export.py` is the
shared exporter; experiment entry points are in `src/experiments/agentic_eval/`.
The evaluation client and vLLM server have separate uv projects/locks because
current Inspect needs huggingface-hub >=1.2 while vLLM 0.11.0 with Transformers
4.57.1 needs huggingface-hub <1. Neither environment modifies probe training.
Install the repository package into the client with `uv pip install --no-deps -e .`
after syncing; then use `uv run --frozen --no-sync` so uv doesn't remove it.

## Trace contract (schema_version 1)

`trajectories.jsonl`: one row per benchmark/sample/epoch. Common fields:
`benchmark`, `sample_id`, `epoch`, `model`, `provenance`, `messages`, `tools`,
`scores`, `error`, `limit`, `export_errors`, `source_log`, `model_usage`.

Messages are dictionaries with `role` and string `content`. Assistant reasoning
is stored separately as `thinking`; tool calls are OpenAI/HF-style dictionaries
with `id`, `type=function`, `function.name`, and **dictionary**
`function.arguments` (what the GPT-OSS HF template expects). Tool responses have
`tool_call_id` and `name`. Tool error metadata is retained. Tools are separate
function schemas, required when rendering a tool-bearing chat template.
Unsupported multimodal/redacted content causes an explicit export error rather
than silent loss. Roles never come from interpreting message content.

Basic retrospective tokenization (CPU-checked on a real retail trace):

```python
import json
from transformers import AutoTokenizer

row = json.loads(open("trajectories.jsonl").readline())
tokenizer = AutoTokenizer.from_pretrained(".cache/models/gpt-oss-20b")
input_ids = tokenizer.apply_chat_template(
    row["messages"], tools=row["tools"], reasoning_effort="high",
    tokenize=True, add_generation_prompt=False,
)
```

This produces a valid HF-formatted conversation; see the replay caveat below
before interpreting its activations as the original generation-time activations.

`generations.jsonl`: one row per evaluated-model call, with Inspect input/output,
tools, generation config, raw API call and error. This is the fallback for exact
per-call context reconstruction; it excludes user-simulator/grader calls.
Keep original `.eval` logs too. They retain simulator calls, events, scores,
metadata and raw provider requests/responses. JSONL is an analysis convenience,
not a replacement for the source log.

**τ² trap:** `sample.messages` is the user simulator's privileged conversation.
Export `sample.store['StoredAgentState:RetailAgent:messages']` instead. The current
Inspect `sample.store_as(..., instance=...)` reader did not recover instance-
namespaced fields in our fixture; direct documented store keys are tested.

**Replay caveat:** a full-history forward pass is a useful retrospective probe
measurement, but not necessarily the original activation context. Native vLLM
0.11.0 GPT-OSS serving uses openai-harmony, not the cached HF Jinja template, and
adds system/developer messages. Templates may discard old reasoning and change
serialization. Preserve both files. The probe experiment uses an explicit standardized retrospective renderer,
`trajectories/replay.py`, preserving all reasoning. It does not claim to recover
serving-time activations; original generation records support future exact replay.
Do not feed these dictionaries directly to the baseline's simplified Harmony
renderer: it expects already-normalized call messages and has no developer role.
Do not infer missing channels from text. Use the probe metadata's exact model
revision, layer, role order and post-attention-layernorm/pre-MLP site.

## Execution and metrics

One B200 server with 8 concurrent retail samples is the initial configuration.
Generation is sequential within each trajectory; vLLM batches across trajectories.
Fixed selection is Python Random(123) over sorted sample IDs. For more parallelism,
`--shard-count N --shard-index I` partitions the same chosen subset without overlap.
Use separate output directories; merge logs explicitly and reject duplicate IDs.
The default 20 tasks, 900-second per-sample clock limit, 16,384 output tokens per
call and 500,000 sample tokens target a sub-hour pilot after queueing/startup.
The initial 4,096/100,000-token pilot hit 68 per-call caps and truncated 13/20
conversations; those budgets were inadequate for high reasoning.
This is a budget, not a measured throughput guarantee. Limits affect comparability.
Inspect token limits count repeated API input/output usage across calls (including
the simulator), not the token length of the final serialized conversation.
Keep scores for every task; don't select only successful/short trajectories.

Retail uses Azure `openai/azure/gpt-4.1` as customer simulator. A live five-token
check confirmed model `gpt-4.1-2025-04-14`; credentials already exported from the
user's shell are inherited by Slurm, never saved to manifests. API version
2024-10-21, chat completions. Retail's score is deterministic (database state and
required communicated info), so no paid grading model is needed.

Export success rates, scored/error counts, Wilson 95% intervals, truncations,
missing IDs and model-call errors. Upstream retail catches some model errors
inside its loop, including SampleLimitExceededError. In the first live run,
13 samples had `sample_limit` events despite `sample.limit is None`. Count nested
limit events and `max_tokens` generation stop reasons, not just top-level flags.
A completed Inspect log alone does not establish a clean run.
Inspect's original aggregate results and usage are also retained in summary.json.
This is a one-trial subset success rate, not pass^k or a full leaderboard result.

## SWE-bench

Full Verified execution is now validated on Slurm with the Apptainer adapter. It
uses writable per-sample root filesystems, isolated networking, no host-workspace
mount and no API credentials in the sandbox. Gold-patch scoring and stock Inspect
Python/bash-session/editor tools passed live checks. Use the separate locked SWE
client (swebench 4.1.0), not the retail client's incompatible swebench 5.0.2.

Docker Hub's anonymous pull quota interrupted the first attempt. The selected
full run uses the same official images via mirror.gcr.io, pinned by OCI index
digest in `swe_bench_full/image_manifest.json`. Manifest availability does not
guarantee blob availability: two missing layers were downloaded from the official
registry, SHA256-verified, and seeded into Apptainer's blob cache. Retry only image
failures before model execution, retaining original attempts and selection reasons.
Malformed generated Harmony headers produce model API failures; the stock
30-message harness also often exhausts its budget. This configuration is not a
reproduction of OpenAI's reported SWE score.

Sources:
- [Inspect τ² implementation](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/tau2)
- [Inspect SWE-bench](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench)
- [Inspect parallelism](https://inspect.aisi.org.uk/parallelism.html)
- [GPT-OSS model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations)

## First measured result

The 20-task high-reasoning retail pilot (Slurm 1448731) completed in 27m01s,
including startup/export, with 9/20 success (45%, Wilson 95%: 25.8–65.8%).
All 20 traces contain tools and tokenize with the HF template. Two were
sample-time-limited, 16 had at least one output-cap hit, and four had failed
model-call events. These are usable flagged exploratory traces, not a clean
leaderboard reproduction. See the [report](../experiment_reports/agentic/tau2_retail_gptoss20b/README.md)
and its small versioned `data/trajectories.jsonl.gz` archive.

## Full runs and probe transfer

The full retail experiment uses all 114 τ² retail tasks, not the original τ-bench. SWE Verified uses all 500 tasks and the separate `agentic_eval/swe` locked environment (swebench 4.1.0; version 5.0.2 fails the installed Inspect scorer import). `trajectories/apptainer.py` supplies writable isolated instances; gold-patch scoring and Inspect Python/bash-session/editor tools passed live validation. Instance-start does not accept `--pwd`; set working directory on exec.

Dolci sampling: `src/experiments/agentic_eval/dolci.py`, 1,000 uniform rows, seed 123, revision pinned in the script/manifest. Preserve function_calls/functions and map environment to tool; do not invent analysis channels.

Probe replay uses the original eager forward, capturing layers 4/8/12/16/20 together. Explicit 16,384-token windows retain every token and reset positions/context only at window boundaries. Full-window peak measured 136.67 GiB on B200. Cached whole-model chunks and query tiling showed numerical drift in BF16/MoE; these are not used in production scoring. Report windowed versus single-window coverage separately. `agentic_probes/report.py` creates README, APPENDIX, metrics and an offline standalone viewer from per-trajectory gzip JSON predictions. Keep raw data/predictions/data-filled HTML ignored; push code and aggregate reports periodically.

## Canonical repairs and offline viewer

Unknown generated tool names are preserved as `__unknown_tool__` with
`name_missing: true` and `original_function`, rather than dropping the history.
`repair_exports.py` writes corrected records into a separate run. `finalize.py`
accepts explicit replacement runs only for export repairs or image-preparation
failures with zero model calls. It recomputes metrics from selected records.
`select_predictions.py` validates selected predictions against canonical messages,
tools, scores, errors and source logs, recording hashes and linking immutable files.

Large offline viewers must keep compressed trajectory payloads in separate inert
script blocks: embedding hundreds of MB in one JavaScript literal exceeds V8's
source-size limit. Decode only the selected trajectory. `viewer_check.py` tests
real files offline with Chromium, including each dataset/layer and longest trace.
Use an isolated Playwright environment so production Python locks stay unchanged.
