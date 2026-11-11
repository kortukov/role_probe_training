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
  reasoned on the analysis channel; the rest deliberated in commentary preambles. The
  ground-truth label of a token is fixed by the channel the model wrote it on, not by
  what the text looks like, so those preambles count as `assistant` even though they read
  like reasoning, and the dominant probe error is preamble text predicted as chain of
  thought. Each pipeline made its own choice about which earlier reasoning
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
recipient in both positions". Second, policies about what to keep: on every request,
vLLM 0.32 looks through the history the client sent, finds the last assistant message on
the final channel, and removes every chain-of-thought message before it (the library's
default only does this when the final answer is the very last message). Nothing is
removed while no final answer exists yet, so reasoning accumulates through a tool loop
and is still present when the final answer is generated; it disappears on the next
request after that answer, which only happens when the user speaks again. vLLM 0.32 also
treats only the first system or developer message as developer
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
| vLLM 0.32 (AppWorld on hx2) | Kept throughout. Inspect sends it back in a `reasoning` field and vLLM renders it on the analysis channel; it would drop reasoning from before a final answer, but an AppWorld episode ends at its first final answer (Inspect's loop stops when the model returns no tool call), so no request ever followed one. Each block was preceded by the blank commentary message. Preambles came back as assistant text. |
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

**One renderer.** `src/role_probe_training/harmony/` wraps the Harmony library. That
library has two names: it is installed as the package `oss-harmony` (the name uv and pip
use, pinned in `pyproject.toml`; it is a fork of OpenAI's `openai-harmony` package), and
Python code imports it as the module `openai_harmony`. It renders the
Harmony system message, the developer message with instructions and tool schemas, user
messages and tool outputs exactly as the library does, keeps all chain of thought in the
prompt, and attaches per-token role labels derived from each message's author and
channel. Header tokens and special tokens are unlabelled. (The special tokens are the
seven reserved vocabulary entries `<|start|>`, `<|channel|>`, `<|constrain|>`,
`<|message|>`, `<|end|>`, `<|call|>` and `<|return|>`. Each is a single token that
tokenising ordinary text can never produce; together with the header text they mark the
structure of the conversation rather than its content.) Settings that are not part of the
conversation but become prompt tokens, the reasoning effort and the system-message date,
are fixed in configuration and recorded with every output together with the library
version. The Hugging Face chat template is no longer used anywhere.

**Model turns are replayed, never re-rendered.** When the server builds a prompt, it
does not rebuild the model's earlier turns from parsed messages. That would be lossy, for
two reasons. First, the model writes its own headers: after the prompt ends in
`<|start|>assistant`, the channel, the recipient and the content type are all tokens the
model generates, and they do not always match the library's form. An audit of raw
completions from the on-policy runs found calls to the Python tool with content type
`code` and no `<|constrain|>` token, and late-turn headers with the recipient written in
two places, which is also the malformed echo vLLM's source mentions. In the hx2 AppWorld
cohort the model even wrote a header fragment, `commentary to=functions.python code<|message|>`,
inside its reasoning. Parsing such a turn into channel, recipient and arguments and
rendering it again produces the library's canonical header, not the tokens the model
wrote. Second, the OpenAI-style message format does not record whether a piece of
assistant text was a preamble on the commentary channel or a final answer, so the channel
cannot be restored from it. Instead, every earlier model turn is inserted as the
token IDs the model generated for it, exactly as recorded. Two details make this fit the
Harmony format. First, the model was prompted with `<|start|>assistant` and wrote the rest
of the header itself, so its recorded tokens begin at `<|channel|>`; the server puts
`<|start|>assistant` back in front of them. Second, the model ends a final answer with
`<|return|>`, but Harmony renders a final answer from an earlier turn with `<|end|>`, so
that one token is replaced; tool calls keep their `<|call|>`. Assistant text is rendered
from a structured message (role, channel, text) by the Harmony library, through this
repository's renderer, only where no model generated it, so there are no tokens to
replay: the reference answers of the No Robots dataset, the Toucan tool-use text, and the
passages used to fit the probes.

**Token records replace re-rendering for the probes.** Every model call is recorded as
`prompt_ids`, `completion_ids`, the message spans of both, and the rendering settings.
The whole model-visible history of an episode is the last call's prompt plus its
completion. Trajectory rows carry that record under `history`; `encode_record` reads it
and refuses rows without one. There is no fallback to re-rendering old trajectories.

**One server for generation.** `harmony/server.py` is an OpenAI-compatible chat server
that renders with the library, samples completion token IDs from vLLM's completions
endpoint, parses with the library's streaming parser, and writes the records. It asserts
on every request that the new prompt extends the previous prompt of the conversation
exactly. Earlier assistant turns are replayed whether they ended in a tool call (keyed by
the call IDs the server issued) or in a text answer the user then replied to (keyed by
conversation and turn number, with the text checked against what was generated), so
multi-turn chats such as the retail benchmark work. One sampling temperature is fixed per
server and every request must use it: every evaluation collection (AppWorld, retail,
SWE-bench) decodes greedily at temperature 0, while on-policy training samples at 1. It
serves two users:

* the on-policy training loop, through per-episode routes with seed validation (the
  former shim's `template_messages`, template patch and vLLM-side parser are gone);
* evaluation collections, through `src/experiments/eval_role_probes/agentic_eval/shared/harmony_proxy.py` between Inspect and vLLM.

## 4. What is affected downstream

**Code paths that changed.** Probe fitting and clean-chat generation
(`src/experiments/gptoss_role_probes/`), Toucan document headers (`prepare.py`), the No
Robots chat evaluation (`chat_eval/norobots/`), agentic probe scoring
(`probe_scoring/run.py`), the trajectory exporter (`inspect_export.py`, which joins the
records onto trajectories by conversation key), the AppWorld, retail and SWE-bench runners
and their Slurm job files (each job starts one Harmony proxy in front of vLLM, runs the
smoke test and the collection through it, and joins the proxy's `requests.jsonl` onto the
export; the shared `smoke.py` and `export.py` take the proxy's settings and file),
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

**Environments and provenance.** `oss-harmony==0.0.12` is pinned in the root project, which
since 2026-11-09 is the one environment for serving, collection, scoring and training
(`agent_docs/ENVIRONMENTS.md`). vLLM runs from the same environment and imports this fork
too, because the upstream `openai-harmony` is excluded. Training manifests hash the new
`harmony/` sources; resuming checkpoints made before this change needs the usual migration
record.

**Left unchanged.** The SWE-bench collection scripts of the training experiments
(`src/experiments/training_against_role_probes/swe_*/run.py` and their Slurm files), which
evaluate trained checkpoints, still give Inspect the address of vLLM's own chat endpoint
and would produce trajectories without `history`. They need the same proxy changes as the
evaluation runners before a trained checkpoint is re-evaluated on SWE-bench. Their
commands already run from the root environment (vLLM 0.27.1), but they still pass vLLM's
tool-call parser flags, which the proxy path does not use.

**KL anchors.** `src/experiments/training_against_role_probes/generate_kl.py` produces the
fixed KL anchor set that every training script reads: 800 Dolci-Instruct prompts as single
user messages, greedy base-model completions of up to 1,536 tokens, exact token IDs in
`experiment_reports/training_against_role_probes/data/kl/anchors.jsonl`. It now renders the
prompts with this repository's renderer and samples token IDs from a vLLM server that
`generate_kl.sbatch` starts, the same way the Harmony proxy samples trajectories. Before
2026-11-08 it rendered them with the Hugging Face chat template inside vLLM's Python API. For
a single user message without tools the template and the library produce identical token IDs
(verified for the same date), so the recorded anchors (the coauthor's job 1501243) already
match the new rendering apart from the date in the system message, which is now the fixed
date that all prompts share.

**Removed on 2026-11-08.** `src/experiments/eval_role_probes/agentic_eval/shared/tokenize_check.py`
counted tokens by re-rendering exported chats with the template; `history` makes it
unnecessary. `src/experiments/training_against_role_probes/on_policy/generation_diagnostic.py`
and its Slurm file replayed the request that crashed the earlier on-policy pilot's Hugging Face
serving shim under synchronous CUDA error reporting. Both are in Git history.

**To remove after the new on-policy run.** The earlier on-policy pipeline served the model
with Hugging Face generation inside the training process and rendered prompts with the chat
template. It consists of `src/role_probe_training/on_policy/serving.py`, `train.py`,
`collect.py` and `colocated.py`, the whole `src/role_probe_training/generation/` package, and
the scripts that import them: everything in
`src/experiments/training_against_role_probes/hf_generation/` and, in
`src/experiments/training_against_role_probes/on_policy/`, the trainer Slurm files, the serving
benchmarks and the serving checks (`checks.py` keeps its KL cases). `on_policy_recovery/serving.py`
replaced that pipeline, but the pipeline trained the appworld2700, appworld270_batch16 and
appworld270_batch64 pilots, whose reports the main training report still cites and whose
checkpoint migration records hash its files. Delete the pipeline and those pilot reports
together once the on-policy model retrained under the new rendering replaces their numbers,
and once no queued job imports the modules (`agent_docs/CURRENT_HANDOFF.md`). Keep
`src/role_probe_training/on_policy/data.py`, `objective.py` and `vllm_worker.py`: the
current trainers import them.

**How to verify.** From the repository root, with `TIKTOKEN_CACHE_DIR=$PWD/.cache/tiktoken`:

```
uv run python -m unittest src/experiments/harmony_rendering/checks.py
(cd src/experiments/harmony_rendering && uv run python -m unittest server_checks.py)
uv run python -m unittest src/experiments/eval_role_probes/agentic_eval/probe_scoring/checks.py
uv run python src/experiments/gptoss_role_probes/checks.py
```

**Verified so far (CPU).** The renderer reproduces the library's conversation rendering
token for token; the Hugging Face tokenizer re-encodes the rendered text to the same IDs;
labels cover exactly the content tokens; the text of a special token written inside a
message, such as a user typing `<|end|>`, stays ordinary text and never becomes the token;
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

In this repository's five-role labels, system and developer content is `system`, user
content is `user`, analysis content is `cot`, commentary and final content (including
tool-call arguments) is `assistant`, and tool output is `tool`. Header tokens are never
labelled. The library's own system message and the tools block of the developer message
are labelled `system` but marked with source index -1 and excluded from accuracy in
trajectory scoring, because no conversation participant wrote them.

## 6. Two role sets: the five roles and one role per header form

Since 2026-11-09 the content tokens can be labelled in either of two role sets. Both are
defined in `src/role_probe_training/harmony/render.py`, and a probe belongs to one of them:
its metadata lists its roles, and every consumer either reads the set from the probe or
refuses a probe of the other set. The header roles label every token the five roles label,
and some that the five roles leave out (below).

* `ROLES`, the five roles of the role-probe paper, behind every number reported so far.
* `HEADER_ROLES`, ten roles, one for each header form of section 5. They are a role set of
  their own: header probes are fitted, scored and reported as ten roles and are not reduced
  to the five.

A header role is decided by three parsed fields of the message, never by the header text,
so the model's own spellings of a header fall into the same role:

| Header role | Author | Channel | Recipient |
|---|---|---|---|
| `system` | system | none | none |
| `developer` | developer | none | none |
| `user` | user | none | none |
| `analysis` | assistant | analysis | none |
| `commentary` | assistant | commentary | none |
| `final` | assistant | final | none |
| `tool_call` | assistant | commentary | set |
| `tool_output` | a tool | commentary | any |
| `analysis_tool_call` | assistant | analysis | set |
| `analysis_tool_output` | a tool | analysis | any |

What the names cover:

* **`commentary` is the text before a call, not the call.** Three header forms use the
  commentary channel: the assistant's preamble, its tool call and the tool's output. The
  role `commentary` is only the first, the message with no recipient. The JSON arguments
  are `tool_call` and the returned text is `tool_output`.
* **`system` is the library's message; `developer` is what a client calls the system
  prompt.** In a served conversation the only text under the `system` header is the fixed
  message of section 5 (about 60 to 70 tokens), and instructions are rendered under
  `developer`, after the library's `# Instructions` heading.
* **Header-role scoring counts every labelled token; five-role scoring skips library-written
  context.** A header role is the header a token sits under, so header-role trajectory
  scoring counts the library's system message as `system` and the tool-schema block of the
  developer message as `developer`. The system message is the same text in every
  conversation, and the schema block is generated from the tool definitions, which matters
  when reading `system` and `developer` recall. Five-role trajectory scoring still excludes
  both, because there `system` means the instructions given to the model.
  `Rendering.score_mask` holds this rule. The clean-chat evaluation of the fitting script
  scores the library's message in both sets.
* **Tool calls and tool outputs are split by channel, not by tool.** Commentary is meant to
  be shown to the user and analysis is not, so a call or an output on the analysis channel
  has its own role (`analysis_tool_call`, `analysis_tool_output`). The library puts the
  built-in tools (`python`, `browser`) there. A function call that the model writes on the
  analysis channel is `analysis_tool_call` too. A call or an output on the final channel or
  without a channel, which the library does not produce, counts with the commentary roles.
* **The analysis tool roles are rare in our trajectories.** No run declares a built-in
  tool, so `analysis_tool_call` occurs only when the model writes a function call on the
  analysis channel. The renderer puts every function output on the commentary channel,
  also after such a call, so `analysis_tool_output` occurs only when the model itself
  writes a tool-output message on the analysis channel (next point), and in fitting wraps.
* **A message the model wrote under another author's header is labelled by that header.**
  Generation stops only at `<|return|>` or `<|call|>`, so after an `<|end|>` the model
  writes the next header itself and can give it any author: a tool output
  (`functions.NAME to=assistant<|channel|>analysis`) or a user turn. The server marks such
  a turn invalid and the episode ends, but the tokens stay in the record. A header role is
  the header a token sits under, so these messages are `analysis_tool_output`,
  `tool_output`, `user` and so on, and they are scored. The five roles leave them
  unlabelled, because the header names one author and the model is the real one.
* **Unlabelled in both sets:** an assistant message with no channel or an unknown one.

Fitting wraps every passage once per role of the chosen set (`render_document`), so a header
probe sees ten renderings of a passage where a five-role probe sees five. The developer
wrap is the library's developer message with the passage as its instructions; the tool-call
wrap is a commentary message to `functions.NAME` with `<|constrain|>json`, ending in
`<|call|>`. `NAME` is the document's function for Toucan documents and `tool` otherwise.
The two analysis tool wraps are the built-in python tool's forms for every document:
`assistant to=python<|channel|>analysis <|constrain|>code` for the call and
`python to=assistant<|channel|>analysis` for the output.

Where each set is supported:

| Code | Role set |
|---|---|
| `gptoss_role_probes/run.py` (fit, clean-chat evaluation, optimizer audit) | `--roles five` (default) or `--roles header`; header outputs carry a `_header` suffix |
| `gptoss_role_probes/header_roles/fit.py` (header-role probes on the C4/Dolma passages, both fitters run to a convergence test) | header roles only |
| `agentic_eval/probe_scoring/run.py` (trajectory scoring) | the set of the probes it is given |
| `agentic_eval/probe_scoring/report.py` and its viewer, `chat_eval/norobots/` | five roles only |
| training against probes (`src/role_probe_training/training/`, `post_training/`) | five roles only; header probes are refused |

Fitted header-role probes and their accuracy: [header-role fits](experiment_reports/gptoss_role_probes/header_roles/README.md).
