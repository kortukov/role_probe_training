"""Small, explicit GPT-OSS renderer with labels from message metadata.

Training headers match upstream utils/role_templates.py. Labels never come
from searching message content for role names or delimiters.
"""

import re

import numpy as np

ROLES = ("system", "user", "cot", "assistant", "tool")
SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\nCurrent date: 2025-12-31\n\n"
    "Reasoning: medium\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)


def role_label(message):
    role = message["role"]
    if role == "assistant":
        channel = message.get("channel", "final")
        if channel == "analysis":
            return "cot"
        if channel not in ("final", "commentary"):
            raise ValueError(f"Unsupported assistant channel: {channel}")
    if role not in ROLES:
        raise ValueError(f"Unsupported role: {role}")
    return role


def render(messages, *, bos=False):
    """Return Harmony text and (start, end, role, message_index) content spans.

    Canonical messages use role/content, optionally channel, recipient, name,
    and ending. `cot` is accepted as a convenient alias for assistant/analysis.
    Literal control-token strings in content are retained, never parsed as labels.
    """
    text = "<|startoftext|>" if bos else ""
    spans = []
    for i, message in enumerate(messages):
        role = role_label(message)
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("Message content must be a string; normalize tool calls first")
        if role in ("system", "user"):
            header = role
        elif role == "tool":
            name = message.get("name", "")
            header = f"functions.{name} to=assistant<|channel|>commentary"
        else:
            channel = "analysis" if role == "cot" else message.get("channel", "final")
            recipient = f" to={message['recipient']}" if message.get("recipient") else ""
            header = f"assistant{recipient}<|channel|>{channel}"
        text += f"<|start|>{header}<|message|>"
        start = len(text)
        text += content
        spans.append((start, len(text), role, i))
        ending = message.get("ending", "<|end|>")
        if ending not in ("<|end|>", "<|return|>", "<|call|>"):
            raise ValueError(f"Unsupported message ending: {ending}")
        text += ending
    return text, spans


def tokenize_messages(tokenizer, messages, *, bos=False):
    """Tokenize once; label only tokens fully contained in known content spans.

    Headers, endings, special tokens, and tokens crossing span boundaries get -1.
    Returns token IDs, role labels, message indices and character offsets.
    """
    text, spans = render(messages, bos=bos)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    labels = np.full(len(ids), -1, dtype=np.int64)
    message_ids = np.full(len(ids), -1, dtype=np.int64)
    special = np.isin(ids, tokenizer.all_special_ids)
    for start, end, role, index in spans:
        mask = (offsets[:, 0] >= start) & (offsets[:, 1] <= end)
        mask &= (offsets[:, 1] > offsets[:, 0]) & ~special
        labels[mask] = ROLES.index(role)
        message_ids[mask] = index
    return {"text": text, "input_ids": ids, "labels": labels,
            "message_ids": message_ids, "offsets": offsets}


def parse_generated(text):
    """Parse a generated assistant continuation starting after an open header.

    For evaluation generation only; ground truth for later trajectories should
    come from their structured messages. Reject incomplete/non-final answers.
    """
    pieces = re.findall(
        r"<\|start\|>assistant(?P<header>.*?)<\|message\|>(?P<content>.*?)"
        r"(?P<ending><\|end\|>|<\|return\|>|<\|call\|>)", text, re.S
    )
    messages = []
    for header, content, ending in pieces:
        if "to=" in header:
            raise ValueError("Unexpected tool call during chat evaluation")
        channel = header.split("<|channel|>")[-1] if "<|channel|>" in header else "final"
        if channel not in ("analysis", "final"):
            raise ValueError(f"Unexpected generated channel: {channel}")
        messages.append({"role": "assistant", "channel": channel,
                         "content": content, "ending": ending})
    if not messages or messages[-1]["channel"] != "final" or messages[-1]["ending"] != "<|return|>":
        raise ValueError("Generation did not finish a final answer")
    return messages
