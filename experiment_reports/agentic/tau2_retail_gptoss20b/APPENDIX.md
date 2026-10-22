# Pilot trace diagnostics

These diagnostics describe truncation, API failures and representation checks; they are not additional task-success metrics. An output-cap hit means a model response reached its per-call generation limit. A retry is another API attempt after a failure.

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


[Artifact locations and reproduction](RUNNING.md).
