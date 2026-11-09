"""Export agent histories and individual generation contexts from Inspect logs.

Keep the original .eval. The chat representation here is for reading and benchmark
scoring; the probes read the token records the Harmony server wrote (`history`), which
are joined onto each trajectory by conversation key.
"""
import hashlib
import json
import math
from pathlib import Path


def conversation_key(messages):
    """Identify a conversation by its leading system/developer text and first user text.

    Computed identically by the Harmony server (from the request body) and by this
    exporter (from the Inspect log), without any Harmony dependency.
    """
    instructions, first_user = [], None
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else message.content
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                              for part in content)
        role = message.get("role") if isinstance(message, dict) else message.role
        if role in ("system", "developer") and first_user is None:
            instructions.append(content or "")
        elif role == "user":
            first_user = content or ""
            break
        else:
            break
    payload = json.dumps([instructions, first_user], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def attach_history(row, records, context_stops=0):
    """Give one trajectory row the token record of its last model call.

    `records` are the Harmony server's request records of this conversation and
    `context_stops` the number of calls the server refused for lack of context (those
    appear as Inspect model events without a record). A conversation without any
    recorded call gets `history: None` and is unscorable for role metrics.
    """
    indices = sorted(r["generation_index"] for r in records)
    if indices != list(range(len(indices))):
        raise ValueError(f"Sample {row['sample_id']}: missing, duplicate or colliding generation records")
    if len(records) + context_stops != row["generation_records"]:
        raise ValueError(f"Sample {row['sample_id']}: {len(records)} server records and {context_stops} context "
                         f"stops but {row['generation_records']} Inspect model events")
    row["history"] = max(records, key=lambda r: r["generation_index"]) if records else None
    row["unlabelled_generated_tokens"] = sum(
        e["content_end"] - e["content_start"] for r in records for e in r["completion_messages"]
        if e["label"] is None)
    return row


def attach_histories(rows, requests_path):
    """Join the Harmony server's records onto exported rows by conversation key."""
    by_key, stops = {}, {}
    with Path(requests_path).open() as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                by_key.setdefault(record["episode_id"], []).append(record)
    stops_path = Path(requests_path).with_name("episode_stops.jsonl")
    if stops_path.exists():
        for line in stops_path.read_text().splitlines():
            if line.strip():
                key = json.loads(line)["episode_id"]
                stops[key] = stops.get(key, 0) + 1
    keys = [conversation_key(row["messages"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Two exported samples share a conversation key; run epochs separately")
    for row, key in zip(rows, keys):
        attach_history(row, by_key.get(key, []), stops.get(key, 0))
    return rows


def dump(value):
    return value.model_dump(mode="json", exclude_none=True) if hasattr(value, "model_dump") else value


def messages_to_chat(messages):
    """HF-style tool calls, explicit reasoning, and tool identities from metadata."""
    result, calls = [], {}
    for original in messages:
        m = dump(original)
        content = m.get("content", "")
        reasoning = []
        if isinstance(content, list):
            text = []
            for block in content:
                if block["type"] == "text":
                    text.append(block["text"])
                elif block["type"] == "reasoning":
                    if block.get("redacted"):
                        raise ValueError("Cannot export redacted reasoning for token probes")
                    reasoning.append(block["reasoning"])
                else:
                    raise ValueError(f"Unsupported content type: {block['type']}")
            content = "".join(text)
        out = {"role": m["role"], "content": content}
        if reasoning:
            out["thinking"] = "\n".join(reasoning)
        if m.get("tool_calls"):
            out["tool_calls"] = []
            for call in m["tool_calls"]:
                calls[call["id"]] = call["function"] or "__unknown_tool__"
                out["tool_calls"].append({"id": call["id"], "type": "function", "function": {
                    "name": call["function"] or "__unknown_tool__", "arguments": call["arguments"]}})
                if not call["function"]:
                    out["tool_calls"][-1].update(name_missing=True, original_function=call["function"])
        if m["role"] == "tool":
            call_id = m.get("tool_call_id")
            name = m.get("function") or calls.get(call_id)
            if not name:
                name = "__unknown_tool__"
            if name == "__unknown_tool__":
                out.update(name_missing=True, original_function=m.get("function"))
            out.update(name=name, tool_call_id=call_id)
            if m.get("error"):
                out["error"] = m["error"]
        result.append(out)
    return result


def agent_messages(sample, benchmark):
    if benchmark == "tau2_retail":
        messages = sample.store.get("StoredAgentState:RetailAgent:messages", [])
        # Never accidentally export the simulator's privileged scenario/prompt.
        if not messages:
            raise ValueError("Missing RetailAgent stored history")
        return messages
    return sample.messages


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-half), min(1, center+half)]


def export_logs(paths, output, benchmark, provenance, sample_sources=None, requests_path=None):
    from inspect_ai.log import read_eval_log
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, requests, seen, log_results = [], [], set(), []
    for path in paths:
        log = read_eval_log(str(path), resolve_attachments="full")
        log_results.append({"path": str(path), "status": log.status,
                            "results": dump(log.results), "stats": dump(log.stats)})
        for sample in log.samples or []:
            if sample_sources is not None and (str(sample.id) not in sample_sources or str(Path(path).resolve()) != str(Path(sample_sources[str(sample.id)]).resolve())):
                continue
            key = (str(sample.id), sample.epoch)
            if key in seen:
                raise ValueError(f"Duplicate sample/epoch across logs: {key}; select one attempt")
            seen.add(key)
            errors = []
            events = [e for e in sample.events if e.event == "model" and e.role not in ("user", "grader")]
            # Also match the evaluated model: don't accidentally retain other subagents.
            events = [e for e in events if e.model == log.eval.model]
            tools = []
            for event in events:
                if event.tools:
                    tools = [{"type": "function", "function": dump(t)} for t in event.tools]
                    break
            try:
                chat = messages_to_chat(agent_messages(sample, benchmark))
            except (ValueError, KeyError) as exc:
                chat = []
                errors.append(str(exc))
            scores = {k: dump(v) for k, v in (sample.scores or {}).items()}
            row = {"schema_version": 1, "benchmark": benchmark, "sample_id": sample.id,
                   "epoch": sample.epoch, "model": log.eval.model, "provenance": provenance,
                   "generation_records": len(events),
                   "messages": chat, "tools": tools, "scores": scores,
                   "error": dump(sample.error), "limit": dump(sample.limit),
                   "limit_events": [dump(e) for e in sample.events if "limit" in e.event],
                   "generation_token_caps": sum(c.stop_reason == "max_tokens" for e in events for c in e.output.choices),
                   "model_context_stops": sum(c.stop_reason == "model_length" for e in events for c in e.output.choices),
                   "model_retries": sum(e.retries or 0 for e in events),
                   "export_errors": errors, "source_log": str(path),
                   "model_usage": {k: dump(v) for k, v in sample.model_usage.items()},
                   "total_time": sample.total_time,
                   "simulator_stop_seen": any(marker in sample.output.completion for marker in
                        ("###STOP###", "###OUT-OF-SCOPE###", "###TRANSFER###")) if benchmark == "tau2_retail" else None,
                   "all_model_errors": [e.error for e in sample.events if e.event == "model" and e.error], "agent_model_errors": [e.error for e in events if e.error]}
            rows.append(row)
            for index, event in enumerate(events):
                requests.append({"schema_version": 1, "benchmark": benchmark, "sample_id": sample.id,
                    "epoch": sample.epoch, "generation_index": index, "model": event.model,
                    "input": [dump(m) for m in event.input], "output": dump(event.output),
                    "tools": [dump(t) for t in event.tools], "config": dump(event.config),
                    "error": event.error, "retries": event.retries, "call": dump(event.call)})
    if requests_path is not None:
        attach_histories(rows, requests_path)
    for name, data in (("trajectories.jsonl", rows), ("generations.jsonl", requests)):
        target = output / name
        temporary = target.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(target)
    scorer_names = sorted({name for row in rows for name in row["scores"]})
    metrics = {}
    for name in scorer_names:
        values = [r["scores"][name]["value"] for r in rows if name in r["scores"] and not r["error"]]
        if not all(v in ("C", "I", 0, 1) for v in values):
            raise ValueError(f"Nonbinary scorer {name}: add an explicit metric adapter")
        successes = sum(v in ("C", 1) for v in values)
        metrics[name] = {"successes": successes, "scored": len(values), "observed_samples": len(rows),
                         "success_rate_scored": successes/len(values) if values else None,
                         "wilson_95_scored": wilson(successes, len(values)),
                         "success_rate_observed_errors_as_failures": successes/len(rows) if rows else None}
    summary = {"benchmark": benchmark, "samples": len(rows), "metrics": metrics,
               "sample_errors": sum(bool(r["error"]) for r in rows),
               "limited_samples": sum(bool(r["limit"]) or bool(r["limit_events"]) for r in rows),
               "export_errors": sum(bool(r["export_errors"]) for r in rows),
               "agent_model_errors": sum(len(r["agent_model_errors"]) for r in rows),
               "generation_token_caps": sum(r["generation_token_caps"] for r in rows),
               "samples_with_generation_caps": sum(r["generation_token_caps"] > 0 for r in rows),
               "model_context_stops": sum(r["model_context_stops"] for r in rows),
               "samples_with_model_context_stops": sum(r["model_context_stops"] > 0 for r in rows),
               "model_retries": sum(r["model_retries"] for r in rows),
               "samples_with_limit_events": sum(bool(r["limit_events"]) for r in rows),
               "simulator_stop_missing": sum(r["simulator_stop_seen"] is False for r in rows),
               "all_model_errors": sum(len(r["all_model_errors"]) for r in rows),
               "generation_records": len(requests), "inspect_logs": log_results,
               "provenance": provenance,
               "expected_samples": len(provenance.get("sample_ids", rows)),
               "missing_sample_ids": [i for i in provenance.get("sample_ids", []) if str(i) not in {str(r["sample_id"]) for r in rows}],
               "trajectory_sha256": hashlib.sha256((output/"trajectories.jsonl").read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
