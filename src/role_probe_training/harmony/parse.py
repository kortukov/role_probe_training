"""Parse generated Harmony completions into messages with token spans and labels."""
from openai_harmony import HarmonyError, Role, StreamableParser, StreamState

from .render import encoding, label_for, tokens

CHANNELS = ("analysis", "commentary", "final")


def _role_name(role):
    if role is None:
        return None
    return role.value if hasattr(role, "value") else str(role).split(".")[-1].lower()


def walk_completion(completion):
    """Split a completion into messages; return (entries, error).

    Positions use turn coordinates: position 0 is the `<|start|>` of the
    `<|start|>assistant` header that preceded the completion in the prompt, so
    completion token i sits at position i + 2. Entries carry role, channel, recipient,
    content_type, content, start, content_start, content_end, end (exclusive, covering
    the end token) and complete. A truncated last message is returned with
    complete=False. `error` is set when the library rejects the token stream; entries
    before the failure are still returned.
    """
    parser = StreamableParser(encoding(), role=Role.ASSISTANT)
    base = len(tokens().assistant_header)
    entries, current, error = [], None, None
    for index, token in enumerate(completion):
        before = parser.state
        if current is None:
            current = {"start": 0 if index == 0 else base + index}
        try:
            parser.process(token)
        except HarmonyError as exc:
            error = f"{type(exc).__name__}: {exc} at completion token {index}"
            break
        after = parser.state
        if before != StreamState.CONTENT and after == StreamState.CONTENT:
            current.update(role=_role_name(parser.current_role), channel=parser.current_channel,
                           recipient=parser.current_recipient, content_type=parser.current_content_type,
                           content_start=base + index + 1)
        if before == StreamState.CONTENT and after != StreamState.CONTENT:
            message = parser.messages[-1]
            current.update(content_end=base + index, end=base + index + 1, complete=True,
                           content="".join(part.text for part in message.content))
            entries.append(_finish(current))
            current = None
    if current is not None and error is None:
        end = base + len(completion)
        if "content_start" not in current:
            current.update(role=_role_name(parser.current_role), channel=parser.current_channel,
                           recipient=parser.current_recipient, content_type=parser.current_content_type,
                           content_start=end)
        current.update(content_end=end, end=end, complete=False, content=parser.current_content or "")
        entries.append(_finish(current))
    return entries, error


def _finish(entry):
    entry.setdefault("role", None)
    entry.setdefault("channel", None)
    entry.setdefault("recipient", None)
    entry.setdefault("content_type", None)
    entry.update(kind="assistant", name=None, source=0,
                 label=label_for(entry["role"], entry["channel"]) if entry["role"] == "assistant" else None)
    return entry


def completion_to_chat(entries, tools, error=None):
    """OpenAI-style assistant message for the client plus validity issues.

    Mirrors the previous serving shim: any issue makes the turn invalid, which the
    client receives as an empty assistant message without tool calls, ending its tool
    loop. The generated tokens are kept in the records regardless.
    """
    allowed = set()
    for tool in tools or []:
        function = tool.get("function", tool)
        allowed.add(function["name"])
    issues = [f"harmony_error: {error}"] if error else []
    reasoning, content, calls = [], [], []
    for entry in entries:
        if entry["role"] != "assistant":
            issues.append("generated_nonassistant_role")
        channel = entry["channel"]
        if channel not in CHANNELS:
            issues.append(f"unsupported_channel: {channel}")
        recipient = entry["recipient"]
        if recipient and recipient.startswith("functions."):
            name = recipient[len("functions."):]
            if name not in allowed:
                issues.append("unadvertised_tool")
            if not entry["complete"]:
                issues.append("incomplete_tool_call")
            calls.append({"type": "function", "function": {"name": name, "arguments": entry["content"]}})
        elif recipient:
            issues.append(f"unsupported_recipient: {recipient}")
        elif channel == "analysis":
            reasoning.append(entry["content"])
        elif channel in CHANNELS:
            content.append(entry["content"])
    if not calls and not "".join(content):
        issues.append("no_action_or_final_content")
    invalid = bool(issues)
    message = {"role": "assistant", "content": "" if invalid else "\n".join(content)}
    if reasoning:
        message["reasoning_content"] = "\n".join(reasoning)
    if calls and not invalid:
        message["tool_calls"] = calls
    return {"message": message, "invalid": invalid, "issues": issues}
