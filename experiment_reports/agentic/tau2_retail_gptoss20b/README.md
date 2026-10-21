# GPT-OSS-20B on τ²-bench retail: trajectory pilot

Status: collection complete; probe evaluation pending. Date: 2026-10-21.

## 1. Motivation & RQ

Can a small, parallel retail evaluation produce useful tool-bearing GPT-OSS
trajectories for later role-probe evaluation, while measuring task success in
under an hour?

## 2. Evaluation setup

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
- Metrics: task success over **all 20 selected tasks**, Wilson 95% interval,
  errors/limits, usage and wall time. No failed or limited tasks are dropped.
  This is one-trial subset success, not pass^k. Probes have not yet been evaluated.
- Published context: GPT-OSS-20B high reasoning scored **54.8% τ-bench retail**
  and **60.7% SWE-bench Verified** in the
  [model-card table](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations).
  τ-bench and τ²-bench differ; simulator, scaffold, sample size and budgets also
  prevent a direct reproduction claim.

## 3. Factual observations

| Run | Output cap / cumulative token budget | Task success | Wilson 95% interval | Sample limits | Evaluation time |
| --- | --- | --- | --- | --- | --- |
| Initial pilot, 1448719 | 4,096 / 100,000 | 3/20 (15%) | 5.2–36.0% | 13 token-limited | 10m40s |
| Main pilot, 1448731 | 16,384 / 500,000 | **9/20 (45%)** | **25.8–65.8%** | 2 time-limited | **25m25s** |

The main Slurm job completed successfully in **27m01s including startup and
export**. Both runs used the same 20 IDs. The second was launched after auditing
the first; this is not a controlled causal experiment on token budgets.

The main run has:

- **20 tool-bearing trajectories**, including 18 with natural simulator stop
  markers and two time-limited conversations (IDs `110`, `111`).
- **144 tool replies**, **307 reasoning-bearing assistant messages**, and
  **333 evaluated-model generation records**. Simulator prompts are excluded
  from the exported agent chats.
- **32 output-cap hits across 16 tasks**; 14 failed model-call events across four
  tasks (`100`, `110`, `111`, `13`) and 26 recorded retries. These include malformed
  Harmony output errors; all are retained. No top-level sample errors or export
  errors were reported. The fully unflagged trajectories are `21`, `47`, `59`;
  this is a trace-quality diagnostic, not a filtered benchmark score.
- All 20 chats passed Hugging Face template tokenization, with retrospective
  lengths **4,557–11,577 tokens**. This template can drop old reasoning; these
  lengths do not measure the original native-vLLM generation contexts.
- GPT-OSS usage: 1,568,910 input and 877,187 output tokens. Azure simulator usage
  reported by Inspect: 110,936 input, 85,888 cached input and 8,133 output tokens.

Inspect's top-level `sample.limit` missed the initial pilot's 13 nested token-limit
stops. The exporter now checks `sample_limit` events as well as per-call stop
reasons; a regression check covers this. The initial 100,000-token limit counted
repeated prompts and simulator calls, not just the final conversation's length.

Artifacts:

- [Reusable trajectories.jsonl.gz](data/trajectories.jsonl.gz) — 20 full message
  records including reasoning, tools, scores, provenance and quality flags; 326 KiB.
- [Main summary](metrics/summary.json), [per-task CSV](metrics/trajectories.csv),
  [trace audit](metrics/trace_audit.json), [tokenization checks](metrics/tokenization.json).
- [Initial pilot summary](metrics/pilot_4096_summary.json) and
  [job history](metrics/jobs.json), including the corrected tool-parser smoke test
  and one production-node CUDA startup failure before inference.
- Raw `.eval` logs, per-generation JSONL and server logs remain under
  `runs/1448731/` on cluster storage; exact paths and hashes are in the manifests.

## 4. Message — interpretation and conclusion

**Yes: Inspect produced reusable, scored tool-bearing trajectories within the
one-hour target.** A simple shared JSONL interface works, with a small adapter
needed to extract τ²'s agent history rather than its top-level simulator history.
Keeping per-generation records also preserves context for later probe analysis.

The main pilot's 45% success rate is a useful rough capability signal, but it does
not establish agreement with the published 54.8% τ-bench result. The interval is
wide, the benchmark differs, and truncation/serving errors remain. The 15% initial
result should be treated as a budget-limited diagnostic. Neither run should be
presented as an unrestricted full-benchmark reproduction.

For initial probe work, all traces and flags are available; do not select only
successful tasks. If clean termination is required, use explicit quality criteria
and report exclusions. Native Harmony context/channel reconstruction must be
validated before claiming original generation-time activations; the current HF
rendering check only establishes that the saved chats can be tokenized.

[How to run this experiment](RUNNING.md)
