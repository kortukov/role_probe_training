"""Token-exact Harmony rendering for GPT-OSS with per-token role labels.

All rendering goes through the oss-harmony library pinned in pyproject.toml (import
name openai_harmony), which is the reference the model and its serving stacks
follow. Two rules keep rendered prompts identical to what the model saw:

* Turns that the model generated are never re-rendered from parsed text. They are
  replayed from their generated token IDs (`{'role': 'assistant', 'tokens': [...]}`).
  Only system, developer, user and tool-output messages, and dataset text that no
  model produced, are rendered from structured content.
* The chat template of the Hugging Face checkpoint is not used anywhere.

Labels come from message metadata only: the author role and channel of each rendered
message. Header and control tokens are unlabelled (-1).
"""
import dataclasses
import importlib.metadata
import os
from pathlib import Path

import numpy as np
from openai_harmony import (Author, DeveloperContent, HarmonyEncodingName, Message, ReasoningEffort,
                            RenderOptions, Role, SystemContent, ToolDescription, load_harmony_encoding)

ROLES = ("system", "user", "cot", "assistant", "tool")
LIBRARY = "oss-harmony"
LIBRARY_VERSION = importlib.metadata.version(LIBRARY)
# The tool-call content type the model was trained with; vLLM 0.32 deliberately
# renders prior calls in an older `commentary json` form instead. We follow the library.
TOOL_CALL_CONTENT_TYPE = "<|constrain|>json"
# Harmony renders a completed final answer of an earlier turn with <|end|>; the model
# itself ends the turn it is generating with <|return|>.
PRIOR_FINAL_ENDING = "<|end|>"
DEFAULT_TOOL_NAME = "tool"
_ENCODING = None


def encoding():
    """The library's GPT-OSS encoding (vocabulary identical to the HF tokenizer)."""
    global _ENCODING
    if _ENCODING is None:
        cache = Path(".cache/tiktoken")
        if "TIKTOKEN_CACHE_DIR" not in os.environ and cache.is_dir():
            os.environ["TIKTOKEN_CACHE_DIR"] = str(cache.resolve())
        _ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _ENCODING


def special(text):
    ids = encoding().encode(text, allowed_special="all")
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one special token")
    return ids[0]


class Tokens:
    """Control-token IDs, resolved once from the encoding."""
    def __init__(self):
        self.start = special("<|start|>")
        self.message = special("<|message|>")
        self.end = special("<|end|>")
        self.ret = special("<|return|>")
        self.call = special("<|call|>")
        self.channel = special("<|channel|>")
        self.constrain = special("<|constrain|>")
        self.assistant_header = encoding().encode("<|start|>assistant", allowed_special="all")
        self.stop = (self.ret, self.call)


_TOKENS = None


def tokens():
    global _TOKENS
    if _TOKENS is None:
        _TOKENS = Tokens()
    return _TOKENS


@dataclasses.dataclass(frozen=True)
class RenderConfig:
    """Inputs of the Harmony system message that are not part of the conversation."""
    reasoning_effort: str = "high"
    conversation_start_date: str = "2025-12-31"

    def system_content(self):
        effort = ReasoningEffort[self.reasoning_effort.upper()]
        return (SystemContent.new().with_reasoning_effort(effort)
                .with_conversation_start_date(self.conversation_start_date))

    def metadata(self):
        return {"library": LIBRARY, "version": LIBRARY_VERSION,
                "reasoning_effort": self.reasoning_effort,
                "conversation_start_date": self.conversation_start_date,
                "auto_drop_analysis": False, "tool_call_content_type": TOOL_CALL_CONTENT_TYPE,
                "prior_final_ending": PRIOR_FINAL_ENDING, "bos": False,
                "model_turns": "replayed from generated token ids"}

    @classmethod
    def from_metadata(cls, metadata):
        if metadata.get("library") != LIBRARY or metadata.get("version") != LIBRARY_VERSION:
            raise ValueError(f"Record rendered with {metadata.get('library')} {metadata.get('version')}; "
                             f"installed {LIBRARY} {LIBRARY_VERSION}")
        return cls(metadata["reasoning_effort"], metadata["conversation_start_date"])


def label_for(role, channel, recipient=None):
    """Role label of content tokens from author role and channel; None when undefined."""
    if role == "assistant":
        if channel == "analysis":
            return "cot"
        if channel in ("commentary", "final"):
            return "assistant"
        return None
    if role in ("system", "developer"):
        return "system"
    if role in ("user", "tool"):
        return role
    return None


@dataclasses.dataclass
class Rendering:
    """Token IDs plus one entry per rendered message with token positions.

    Entry keys: role, channel, recipient, content_type, name, kind, source, label,
    start, content_start, content_end, end, complete. `source` is the index of the
    input turn (or -1 for context no turn authored: the Harmony system message and the
    tool-schema block of the developer message). Positions are token indices into
    `ids`; `end` is exclusive and covers the message's end token.
    """
    ids: list
    messages: list

    @property
    def labels(self):
        labels = np.full(len(self.ids), -1, dtype=np.int64)
        for entry in self.messages:
            if entry["label"] is not None:
                labels[entry["content_start"]:entry["content_end"]] = ROLES.index(entry["label"])
        return labels

    @property
    def message_ids(self):
        out = np.full(len(self.ids), -1, dtype=np.int64)
        for index, entry in enumerate(self.messages):
            out[entry["content_start"]:entry["content_end"]] = index
        return out

    @property
    def source_message_indices(self):
        return [entry["source"] for entry in self.messages]

    def encoded(self):
        return {"input_ids": np.asarray(self.ids, dtype=np.int64), "labels": self.labels,
                "message_ids": self.message_ids}

    def text_and_offsets(self):
        """Decoded text and per-token character offsets, exact under multi-byte tokens.

        Tokens that only complete part of a UTF-8 sequence get an empty span; the token
        that completes the character gets the whole character.
        """
        enc = encoding()
        pieces, offsets, pending, position = [], [], [], 0
        for token in self.ids:
            pending.append(token)
            piece = enc.decode(pending)
            if "�" in piece and len(pending) < 4:
                offsets.append((position, position))
                continue
            pieces.append(piece)
            offsets.append((position, position + len(piece)))
            position += len(piece)
            pending = []
        if pending:
            piece = enc.decode(pending)
            pieces.append(piece)
            offsets[-1] = (position, position + len(piece))
        return "".join(pieces), np.asarray(offsets, dtype=np.int64)

    def extend(self, other, source=None):
        base = len(self.ids)
        self.ids.extend(other.ids)
        for entry in other.messages:
            shifted = dict(entry)
            for key in ("start", "content_start", "content_end", "end"):
                shifted[key] += base
            if source is not None:
                shifted["source"] = source
            self.messages.append(shifted)
        return self


def _entry(ids, offset, *, role, kind, source, channel=None, recipient=None, content_type=None,
           name=None, label=None):
    """Describe one library-rendered message `<|start|>header<|message|>content END`."""
    control = tokens()
    try:
        message_index = ids.index(control.message)
    except ValueError as error:
        raise ValueError("Rendered message has no <|message|> token") from error
    return {"role": role, "channel": channel, "recipient": recipient, "content_type": content_type,
            "name": name, "kind": kind, "source": source, "label": label,
            "start": offset, "content_start": offset + message_index + 1,
            "content_end": offset + len(ids) - 1, "end": offset + len(ids), "complete": True}


def _message(role, content, *, channel=None, recipient=None, content_type=None, name=None):
    author = Author.new(role, name) if name else None
    message = (Message.from_author_and_content(author, content) if author
               else Message.from_role_and_content(role, content))
    if channel:
        message = message.with_channel(channel)
    if recipient:
        message = message.with_recipient(recipient)
    if content_type:
        message = message.with_content_type(content_type)
    return message


def render_document(role, text, header=None):
    """Wrap one text in one role for probe fitting (no system message, no BOS).

    `header` may carry the Harmony fields of the message's native role: channel,
    recipient, content_type for assistant text; name for tool output.
    """
    header = dict(header or {})
    if not isinstance(text, str):
        raise TypeError("Document text must be a string")
    if role == "system":
        message = _message(Role.SYSTEM, text)
        fields = dict(role="system", kind="system", label="system")
    elif role == "user":
        message = _message(Role.USER, text)
        fields = dict(role="user", kind="user", label="user")
    elif role == "cot":
        message = _message(Role.ASSISTANT, text, channel="analysis")
        fields = dict(role="assistant", kind="assistant", channel="analysis", label="cot")
    elif role == "assistant":
        channel = header.get("channel", "final")
        recipient = header.get("recipient")
        content_type = header.get("content_type", TOOL_CALL_CONTENT_TYPE if recipient else None)
        if label_for("assistant", channel) is None and channel != "analysis":
            raise ValueError(f"Unsupported assistant channel {channel!r}")
        message = _message(Role.ASSISTANT, text, channel=channel, recipient=recipient, content_type=content_type)
        fields = dict(role="assistant", kind="assistant", channel=channel, recipient=recipient,
                      content_type=content_type, label=label_for("assistant", channel))
    elif role == "tool":
        name = "functions." + (header.get("name") or DEFAULT_TOOL_NAME)
        message = _message(Role.TOOL, text, channel="commentary", recipient="assistant", name=name)
        fields = dict(role="tool", kind="tool", channel="commentary", recipient="assistant", name=name, label="tool")
    else:
        raise ValueError(f"Unsupported role: {role}")
    ids = list(encoding().render(message))
    return Rendering(ids, [_entry(ids, 0, source=0, **fields)])


def _tool_descriptions(tools):
    descriptions = []
    for tool in tools or []:
        function = tool.get("function", tool) if isinstance(tool, dict) else tool
        descriptions.append(ToolDescription.new(function["name"], function.get("description") or "",
                                                parameters=function.get("parameters")))
    return descriptions


def _developer(instructions, descriptions, source):
    """Developer message; its tool-schema block becomes a separate `tools` entry."""
    content = DeveloperContent.new()
    if instructions:
        content = content.with_instructions(instructions)
    if descriptions:
        content = content.with_function_tools(descriptions)
    ids = list(encoding().render(Message.from_role_and_content(Role.DEVELOPER, content)))
    entry = _entry(ids, 0, role="developer", kind="developer", source=source, label="system")
    if not descriptions:
        return ids, [entry]
    if not instructions:
        entry.update(kind="tools", source=-1)
        return ids, [entry]
    plain = list(encoding().render(Message.from_role_and_content(
        Role.DEVELOPER, DeveloperContent.new().with_instructions(instructions))))
    split = 0
    for split, (a, b) in enumerate(zip(ids, plain[:-1])):
        if a != b:
            break
    else:
        split = len(plain) - 1
    tools_entry = dict(entry, kind="tools", source=-1, start=split, content_start=split)
    entry.update(content_end=split, end=split)
    return ids, [entry, tools_entry]


def render_turns(turns, config, tools=None, *, system=True):
    """Render a conversation the way the model is served: Harmony system message,
    developer message with instructions and tool schemas, then the turns.

    Turn dicts: {'role': 'developer'|'system', 'content'} (leading only; both mean the
    developer instructions, as vLLM maps OpenAI system messages), {'role': 'user',
    'content'}, {'role': 'tool', 'name', 'content'}, {'role': 'assistant', 'tokens'}
    for generated turns, or {'role': 'assistant', 'content', 'channel', 'recipient',
    'content_type'} for dataset text no model generated.
    """
    enc = encoding()
    descriptions = _tool_descriptions(tools)
    rendering = Rendering([], [])
    index = 0
    instructions, sources = [], []
    while index < len(turns) and turns[index]["role"] in ("system", "developer"):
        instructions.append(turns[index]["content"])
        sources.append(index)
        index += 1
    if system:
        options = RenderOptions(conversation_has_function_tools=bool(descriptions))
        ids = list(enc.render(Message.from_role_and_content(Role.SYSTEM, config.system_content()), options))
        rendering.extend(Rendering(ids, [_entry(ids, 0, role="system", kind="system", source=-1, label="system")]))
    if instructions or descriptions:
        ids, entries = _developer("\n\n".join(instructions), descriptions, sources[0] if sources else -1)
        rendering.extend(Rendering(ids, entries))
    for position in range(index, len(turns)):
        turn = turns[position]
        role = turn["role"]
        if role in ("system", "developer"):
            raise ValueError("System or developer messages are only supported at the start")
        if role == "user":
            ids = list(enc.render(_message(Role.USER, _text(turn["content"]))))
            rendering.extend(Rendering(ids, [_entry(ids, 0, role="user", kind="user", source=position, label="user")]))
        elif role == "tool":
            name = "functions." + (turn.get("name") or DEFAULT_TOOL_NAME)
            ids = list(enc.render(_message(Role.TOOL, _text(turn["content"]), channel="commentary",
                                           recipient="assistant", name=name)))
            rendering.extend(Rendering(ids, [_entry(ids, 0, role="tool", kind="tool", source=position, label="tool",
                                                    channel="commentary", recipient="assistant", name=name)]))
        elif role == "assistant":
            if "tokens" in turn:
                rendering.extend(replay_turn(turn["tokens"]), source=position)
            else:
                rendering.extend(render_document("assistant", turn["content"], turn), source=position)
        else:
            raise ValueError(f"Unsupported turn role {role!r}")
    return rendering


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                raise ValueError(f"Unsupported content part: {part!r}")
        return "".join(parts)
    raise TypeError(f"Unsupported message content {type(content).__name__}")


def replay_turn(completion):
    """A previously generated assistant turn, as the model sees it in a later prompt.

    The tokens are the generated completion (everything after `<|start|>assistant`,
    including the stop token). A final answer's <|return|> becomes <|end|> as in every
    Harmony rendering of earlier turns; tool calls keep <|call|>. Labels come from the
    parsed message boundaries and channels.
    """
    from .parse import walk_completion
    control = tokens()
    completion = list(completion)
    if not completion or completion[-1] not in control.stop:
        raise ValueError("Generated turn does not end with <|return|> or <|call|>; cannot be replayed")
    entries, error = walk_completion(completion)
    if error is not None:
        raise ValueError(f"Generated turn is not parseable Harmony: {error}")
    ids = list(control.assistant_header) + completion
    if ids[-1] == control.ret:
        ids[-1] = control.end
    return Rendering(ids, [dict(entry, source=0) for entry in entries])


def completion_prompt(rendering):
    """Prompt token IDs that ask the model for its next turn."""
    return list(rendering.ids) + list(tokens().assistant_header)
