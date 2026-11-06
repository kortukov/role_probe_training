# One renderer for GPT-OSS prompts: why, what differed, what changed

This note explains a change to how this repository turns conversations into the token
sequences that GPT-OSS-20B reads. It is written for a reader with no prior context.

## 1. The problem: the same conversation became different tokens in different places

GPT-OSS only ever reads and writes tokens. Our data lives as structured messages
(role, channel, text). Converting messages into tokens is called *rendering*; converting
the model's generated tokens back into messages is called *parsing*. The model was
trained on one specific rendering convention, called Harmony, in which every message has
a header such as `<|start|>assistant<|channel|>analysis<|message|>` and an end token.

Until now, four different pieces of code rendered our conversations, and they did not
agree:

| Where | Renderer | Used for |
|---|---|---|
| vLLM 0.32 chat endpoint | the Harmony library plus vLLM's own rewrites | AppWorld evaluation collections on hx2 |
| vLLM 0.11 chat endpoint | an older Harmony library (0.0.8) | retail and SWE-bench evaluation collections |
| the on-policy training shim (`on_policy_recovery/serving.py`) | the Hugging Face chat template of the checkpoint | prompts during on-policy training |
| this repository (`probes/harmony.py`, `trajectories/replay.py`) | a hand-written renderer, written in this repository rather than taken from a library | probe fitting, clean-chat evaluation, and replaying recorded trajectories for probe scoring |

Why vLLM did the rendering for the evaluation trajectories. The AppWorld, retail and
SWE-bench trajectories are produced by Inspect, an agent benchmark harness that talks to
the model only through the OpenAI chat-completions API: it sends messages and tool
definitions as JSON and receives a parsed assistant message back. It never handles
tokens. The inference server, vLLM, does the rendering, the generation and the parsing,
so at every step vLLM chose the exact prompt tokens the model read. It also parsed the
model's own earlier turns into JSON and rendered them again on the next request, so even
those reached the model in vLLM's form rather than as the tokens it had generated.
Nothing in this repository saw those tokens: Inspect logged only the parsed messages, and
the replay code rebuilt tokens from them with the hand-written renderer.

Why this matters for the probes. A role probe reads the model's internal activations at
each token and predicts which role (system, user, chain of thought, assistant, tool)
wrote that token. Activations depend on the whole token sequence that precedes a token,
including the header tokens. So:

* **Training and evaluation were mismatched.** The probes were fitted on texts wrapped in
  the headers written by this repository's hand-written renderer, then evaluated on trajectories replayed by the same
  hand-written renderer, while the model had generated those trajectories under vLLM's
  rendering. The trajectory tokens fed to the probes were never the tokens the model had
  actually seen when it generated the text.
* **On-policy training ran the model on the HF chat template which is different from Harmony.** The Hugging Face
  chat template differs from the Harmony specification (see the public discussion
  "Errors in chat template compared to spec" on the gpt-oss-120b model page). It renders a
  commentary preamble back to the model as chain of thought, quotes tool outputs as JSON
  strings, raises an exception for a turn that has reasoning, a preamble and a tool call,
  omits the `<|constrain|>` token, and stamps every prompt with the current calendar day.
  A model served through it does not see conversations in the form it was trained on, so
  it may not function as intended, and any training signal derived from those contexts
  inherits the defects.
* **Earlier reasoning was in the prompt in some pipelines and not in others.** The
  probes need the chain-of-thought tokens for three reasons. Chain of thought is one of
  the five classes the probe predicts (text on the Harmony analysis channel is labelled
  `cot`) and every results table reports a reasoning score, so when those tokens are
  absent that class cannot be scored and the overall accuracy is computed over a
  different mix of tokens. Training against the probes scores the model's reasoning
  tokens directly. And because the activation at a token depends on every token before
  it, whether a reasoning block precedes a tool call or a final answer also changes the
  activations the probe reads on the assistant and tool tokens. The boundary is also
  where the probes struggle most: in the hx2 AppWorld cohort only 21 percent of calls
  reasoned on the analysis channel, the rest deliberated in commentary preambles that
  are correctly labelled assistant, and the dominant error is assistant text predicted
  as chain of thought. Each pipeline made its own choice about which earlier reasoning
  to keep: vLLM 0.32 kept it inside a tool loop, vLLM 0.11 dropped all of it, the
  Hugging Face template kept it only within the current tool-call chain, and the
  hand-written replay always inserted every block. Section 2 gives the exact rules.

## 2. How the renderings differed, on one example

The conversation: a developer instruction, a user question, one model turn with chain of
thought, a short preamble and a tool call, the tool's output, and a final answer.

Element by element:

| Element | Harmony library (oss-harmony 0.0.12) | vLLM 0.32 serving | Hugging Face chat template | this repository's former hand-written renderer |
|---|---|---|---|---|
| Preamble before a call | `commentary` channel | `commentary` | moved to the `analysis` channel | `commentary` |
| Tool-call header | `assistant<\|channel\|>commentary to=functions.get_weather <\|constrain\|>json` | `assistant to=functions.get_weather<\|channel\|>commentary json` (rewritten to the pre-0.0.10 form) | same as vLLM | `assistant to=functions.get_weather<\|channel\|>commentary` (no content type) |
| Tool-output header | `functions.get_weather<\|channel\|>commentary to=assistant` | same | `functions.get_weather to=assistant<\|channel\|>commentary` | same as the template |
| Tool-output body | the raw output text | raw text | the text quoted as a JSON string, with escaped quotes | raw text |
| Earlier chain of thought | dropped unless `auto_drop_analysis=False` | kept inside a tool loop, dropped before the last final | dropped once a later final exists | always kept |
| Ending of an earlier final answer | `<\|end\|>` | `<\|end\|>` | `<\|end\|>` | `<\|return\|>` |
| Date in the system message | caller's choice | caller's choice | today's date from the clock | fixed `2025-12-31` |
| A turn with reasoning, preamble and call | rendered | rendered, in the order commentary, analysis, call | exception | rendered |

The model itself writes tool-call headers in the library's form (recipient after
`<|channel|>`, with `<|constrain|>json`), which the Hugging Face maintainers confirm they
tried to match. There is therefore one reference: the Harmony library. We pin its
version, because the header form changed at library version 0.0.10 and may change again.
Section 5 lists every header form the library produces.

Why vLLM's rendering differs from the library it uses. Three causes. First, deliberate
rewrites: vLLM's source writes the tool-call content type as plain `json` because, in
its words, "there is not clear evidence that improves accuracy over `json` and some
anecdotes to the contrary", and it rewrites earlier tool-call headers back to the
pre-0.0.10 form because "GPT-OSS can echo the latter as a malformed header with the
recipient in both positions". Second, policies about what to keep: vLLM 0.32 removes
every chain-of-thought message before the last final answer, which goes beyond the
library's default, treats only the first system or developer message as developer
instructions, and renders the newline that Inspect sends as content on tool-call turns
as a blank commentary message; vLLM 0.11 kept only the tool calls of earlier assistant
turns. Third, library version drift: the tool-call header changed at oss-harmony 0.0.10,
vLLM 0.11 shipped 0.0.8, and the hand-written renderer used the old slot order with no
content type. Underneath all three, the chat-completions API carries content, reasoning
and tool-call fields rather than Harmony headers, so every server reconstructs the
headers, and each one did so differently. That is why model turns are now replayed from
their generated tokens instead of being re-rendered.

Which earlier chain of thought was in the prompt. In an agent episode each model turn has
optional reasoning, an optional preamble and a tool call, then the tool result, with a
final answer only at the end. At each new request:

| Pipeline | Earlier reasoning in the prompt |
|---|---|
| vLLM 0.32 (AppWorld on hx2) | Kept. Inspect sends it back in a `reasoning` field, vLLM renders it on the analysis channel and drops only analysis before the last final answer, which inside an episode does not exist yet. Each block was preceded by the blank commentary message. Preambles came back as assistant text. |
| vLLM 0.11 (retail, SWE-bench) | Dropped entirely, together with the preambles. From the second turn on, the model never saw its own earlier reasoning. |
| Hugging Face template (on-policy shim) | Kept within the current tool-call chain, dropped once a later turn has no tool call. A preamble sent as content next to a tool call was rendered on the analysis channel, so assistant text came back as chain of thought. |
| Hand-written renderer (old replay for probe scoring) | Always inserted: every logged `thinking` field became an analysis message, whatever the model had seen. |
| New renderer | Kept (`auto_drop_analysis=False`), and model turns are their own generated tokens, so the prompt is what the model saw. |

The concrete consequence: a retail or SWE-bench trajectory was generated without any
earlier reasoning in context and then scored with all of it inserted, so the probe read
reasoning tokens inside a context the model never had. The on-policy model was trained on
contexts where its own preambles had been relabelled as reasoning.

Rendered text of the model turn, for the two conventions that matter most:

```
Harmony library (what the model generates and what we now render):
<|start|>assistant<|channel|>analysis<|message|>Need the weather tool.<|end|>
<|start|>assistant<|channel|>commentary<|message|>**Plan** call the tool.<|end|>
<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}<|call|>
<|start|>functions.get_weather<|channel|>commentary to=assistant<|message|>{"temperature": 20}<|end|>

Hugging Face chat template (what the on-policy shim fed back):
<|start|>assistant<|channel|>analysis<|message|>**Plan** call the tool.<|end|>
<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city":"Paris"}<|call|>
<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>"{\"temperature\": 20}"<|end|>
```

In the template version the chain of thought is gone, the preamble has become chain of
thought, and the tool output is a quoted string.

## 3. What the change does

**One renderer.** `src/role_probe_training/harmony/` wraps the oss-harmony library
(version pinned in `pyproject.toml`, import name `openai_harmony`). It renders the
Harmony system message, the developer message with instructions and tool schemas, user
messages and tool outputs exactly as the library does, keeps all chain of thought in the
prompt, and attaches per-token role labels derived from each message's author and
channel. Header and control tokens are unlabelled. Settings that are not part of the
conversation but become prompt tokens, the reasoning effort and the system-message date,
are fixed in configuration and recorded with every output together with the library
version. The Hugging Face chat template is no longer used anywhere.

**Model turns are replayed, never re-rendered.** Parsing a generated turn and rendering
it again is lossy: the model does not always emit headers in the library's exact form,
and the OpenAI-style message format loses the channel of a preamble. So an earlier
assistant turn in a prompt is the model's own completion tokens, verbatim, preceded by
`<|start|>assistant`; only a final answer's `<|return|>` becomes `<|end|>`, as in every
Harmony rendering of earlier turns. Structured assistant text is rendered only when no
model produced it (dataset answers, Toucan tool-use text, probe-fitting documents).

**Token records replace re-rendering for the probes.** Every model call is recorded as
`prompt_ids`, `completion_ids`, the message spans of both, and the rendering settings.
The whole model-visible history of an episode is the last call's prompt plus its
completion. Trajectory rows carry that record under `history`; `encode_record` reads it
and refuses rows without one. There is no fallback to re-rendering old trajectories.

**One server for generation.** `harmony/server.py` is an OpenAI-compatible chat server
that renders with the library, samples completion token IDs from vLLM's completions
endpoint, parses with the library's streaming parser, and writes the records. It asserts
on every request that the new prompt extends the previous prompt of the conversation
exactly. It serves two users:

* the on-policy training loop, through per-episode routes with seed validation (the
  former shim's `template_messages`, template patch and vLLM-side parser are gone);
* evaluation collections, through `shared/harmony_proxy.py` between Inspect and vLLM.

## 4. What is affected downstream

**Code paths that changed.** Probe fitting and clean-chat generation
(`src/experiments/gptoss_role_probes/`), Toucan document headers (`prepare.py`), the No
Robots chat evaluation (`chat_eval/norobots/`), agentic probe scoring
(`probe_scoring/run.py`), the trajectory exporter (`inspect_export.py`, which joins the
records onto trajectories by conversation key), the AppWorld runner and its hx2 launcher,
the on-policy recovery module (`serving.py`, `collection.py`, `collect.py`,
`colocated.py`, `train.py`), and the training data helpers (`trajectory_examples(path)`
and `probe_examples(path)` no longer take a tokenizer). The hand-written renderer
`probes/harmony.py` was deleted; `ROLES` now lives in `role_probe_training.harmony`.

**Data that must be regenerated.** Every recorded trajectory without token records:
the retail, SWE-bench and AppWorld evaluation cohorts, the on-policy training and
validation trajectories, the clean-chat conversations and the No Robots generations.
Retail additionally needs the Azure customer-simulator credentials. Until regenerated,
the existing reports describe the old rendering and should be read as such; their code
is in Git history at the commit before this change.

**Probe weights.** Fitting documents are now wrapped in the library's headers, which
differ from the old hand-written ones in the tool-call and tool-output headers. Existing
probe weights were fitted under the old headers; re-score them under the new rendering
before refitting.

**Contexts the model sees in new runs.** Tool-call headers carry `<|constrain|>json`
(vLLM avoids this because the model sometimes echoes the form badly; the server's parser
events now measure that instead of assuming it), preambles stay on the commentary
channel, earlier chain of thought stays in the prompt, the blank assistant message that
Inspect used to inject is gone, and the system-message date is fixed. New on-policy
results are not directly comparable with earlier runs.

**Environments and provenance.** `oss-harmony==0.0.12` was added to the root project and
to the training runtime lock. Training manifests hash the new `harmony/` sources;
resuming checkpoints made before this change needs the usual migration record.

**Left unchanged.** The retail and SWE-bench runners still point Inspect at vLLM's chat
endpoint and need the same proxy before their cohorts are regenerated. Diagnostics that
call the Hugging Face template (`shared/tokenize_check.py`, `generate_kl.py`,
`on_policy/generation_diagnostic.py`) and the archived shims (`on_policy/serving.py`,
`generation/serving.py`) are not on any current path and were not touched.

**How to verify.** From the repository root, with `TIKTOKEN_CACHE_DIR=$PWD/.cache/tiktoken`:

```
uv run python -m unittest src/experiments/harmony_rendering/checks.py
(cd src/experiments/harmony_rendering && uv run python -m unittest server_checks.py)
uv run python -m unittest src/experiments/eval_role_probes/agentic_eval/probe_scoring/checks.py
uv run python src/experiments/gptoss_role_probes/checks.py
```

**Verified so far (CPU).** The renderer reproduces the library's conversation rendering
token for token; the Hugging Face tokenizer re-encodes the rendered text to the same IDs;
labels cover exactly the content tokens; control-token strings inside content stay text;
the prefix property holds across turns; the server round-trips a two-turn tool
conversation with a fake sampler, validates seeds and routes, and refuses an assistant
turn it did not generate. Still to do: a one-episode run against a live vLLM server, the
regeneration of the cohorts, and the report updates that follow.

## 5. Reference: every header form in oss-harmony 0.0.12

Every message is `<|start|>` + header + `<|message|>` + content + ending. The header has
up to four parts: the author, an optional recipient (`to=NAME`), an optional channel
(`<|channel|>NAME`) and an optional content type (` <|constrain|>TYPE`). The tokens in
angle brackets are single special tokens; author names, recipient names, channel names
and type names are ordinary text tokens. Rendered with the library (verified by rendering
each case):

```
System and developer messages
<|start|>system<|message|>...<|end|>
<|start|>developer<|message|>...<|end|>

User messages
<|start|>user<|message|>...<|end|>
<|start|>user:NAME<|message|>...<|end|>                       (a user with a name; not used here)

Assistant messages, one per channel
<|start|>assistant<|channel|>analysis<|message|>...<|end|>     (chain of thought)
<|start|>assistant<|channel|>commentary<|message|>...<|end|>   (preamble shown to the user)
<|start|>assistant<|channel|>final<|message|>...<|end|>        (final answer, <|return|> when the model writes it)
<|start|>assistant<|message|>...<|end|>                        (no channel; legal for the renderer, but the
                                                               system message tells the model a channel is required)

Assistant tool calls
<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>...<|call|>
<|start|>assistant to=python<|channel|>analysis <|constrain|>code<|message|>...<|call|>         (built-in tool; not used here)
<|start|>assistant to=browser.search<|channel|>analysis <|constrain|>json<|message|>...<|call|>  (built-in tool; not used here)

Tool outputs (the author is the tool's name, never the word "tool")
<|start|>functions.NAME<|channel|>commentary to=assistant<|message|>...<|end|>
<|start|>python to=assistant<|channel|>analysis<|message|>...<|end|>                             (built-in tool; not used here)

Generation prompt (the model completes the header itself, starting at <|channel|>)
<|start|>assistant
```

Rules the library applies, which explain the shapes above:

* **Recipient position depends on the channel.** On the `commentary` channel the recipient
  follows the channel (`assistant<|channel|>commentary to=functions.NAME`). On any other
  channel, or without a channel, it precedes the channel (`assistant to=python<|channel|>analysis`).
  The same rule applies to tool outputs, which is why a function output reads
  `functions.NAME<|channel|>commentary to=assistant`. This is the change made at library
  version 0.0.10; before it the recipient always came first and the content type was
  plain text (`commentary json`) without the `<|constrain|>` token.
* **Content type is normalised.** A content type given as `json` or `code` is rendered as
  `<|constrain|>json` or `<|constrain|>code`; the renderer always emits the special token.
  vLLM's plain `json` form is therefore not something the library can produce.
* **Endings.** Every completed message in a prompt ends with `<|end|>`. A message addressed
  to a tool ends with `<|call|>`. The model ends its own final answer with `<|return|>`;
  when that answer is fed back as an earlier turn, the ending becomes `<|end|>`. The two
  tokens that stop generation are `<|call|>` and `<|return|>`.
* **Tool authors must be named.** Rendering a tool message without a name raises
  "Tools should have a name!". This repository renders a tool output under
  `functions.NAME` from the recorded tool name, and under `functions.tool` when no name
  was recorded.

The system message the library writes is also fixed text, and it is what tells the model
about channels:

```
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-12-31

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.
```

The last line appears only when function tools are declared; the date and reasoning level
are the configured values recorded with every output.

In this repository's labels, system and developer content is `system`, user content is
`user`, analysis content is `cot`, commentary and final content (including tool-call
arguments) is `assistant`, and tool output is `tool`. Header tokens are never labelled.
The library's own system message and the tools block of the developer message are
labelled `system` but marked with source index -1 and excluded from accuracy, because no
conversation participant wrote them.
