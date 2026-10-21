# Shared agentic trajectory and probe schema

A trajectory is one JSONL row (`schema_version: 1`) with `benchmark`, `sample_id`,
`epoch`, `model`, `messages`, `tools`, `scores`, `error`, `export_errors`, and
`provenance`. A message has a role and string content; optional fields retain
`thinking`, `channel`, `tool_calls`, `tool_call_id`, `name`, and source-specific
function metadata. Inspect exports keep the evaluated agent's messages, not the
τ² customer simulator's privileged context. Original Inspect logs and per-call
contexts remain available separately.

Dolci's `environment` role maps explicitly to `tool`, with `source_role` retained.
Its aggregate tool replies have `name: environment`, because one reply can contain
multiple function outputs and the dataset supplies no reliable unique call ID.
`function_calls` remains the original string; `functions` remains the original
JSON schema string. No code is executed while normalizing these fields.
No analysis channel is inferred from the prose. The source dataset revision and
uniform sampled row indices are recorded in the manifest.

`replay.py` produces the same explicit Harmony representation for all sources.
It expands structured thinking into assistant/analysis, function calls into
assistant/commentary directed to the named function, and replies into tool
messages. Dolci's unparsed function-call strings use its documented XML wrapper.
Tool schemas are synthetic system context with source message index -1. Their
tokens are predicted but excluded from accuracy. Original system messages are
included in accuracy. This representation preserves retrospective reasoning;
it does not claim to reconstruct each original serving prompt (the native
GPT-OSS serving formatter may omit historical reasoning).

Probe predictions use one gzip-compressed JSON object per trajectory:

- Original `trajectory`, `rendered_messages`, and `source_message_indices`.
- `text`, `input_ids`, character `offsets`, integer `labels`, `message_ids`,
  and `score_mask` for the full replay, including excluded structural tokens.
- `roles`: ordered probability/label classes. A label of -1 is unscored.
- `context_window_starts`: explicit 16,384-token context/position resets.
  Each window receives one original eager forward, with all requested pre-MLP
  hooks attached. No token sampling, token truncation, or cached model chunks.
- `layers`: keyed by zero-based layer, with per-token `predictions` and
  `probabilities`, a content-token confusion matrix, counts and argmax accuracy.

All windows use the existing GPT-OSS checkpoint dequantized to BF16, matching
probe training. A 16K full window peaked at 136.67 GiB on B200 in the hardware
smoke test. Trajectories longer than this have incomplete prior context after
a boundary; report their coverage and single-window results separately.

`src/experiments/agentic_probes/report.py` consumes directories of these records
and emits aggregate metrics, a main report, an appendix of confusion matrices,
and a standalone HTML viewer. Add datasets via repeated `--dataset NAME=PATH`.
The viewer stores each gzip/base64 record in its own inert HTML data block (avoiding JavaScript source-size limits), decompresses one trajectory on
selection, and pages token spans. It inserts untrusted trajectory text using
DOM text nodes, never executable HTML, and needs no network requests.
