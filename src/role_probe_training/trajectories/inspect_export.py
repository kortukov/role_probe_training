"""Export agent histories and individual generation contexts from Inspect logs.

Keep the original .eval: this normalized chat representation is not a claim of
byte-identical serving prompts. In particular Harmony may discard old reasoning.
"""
import hashlib
import json
import math
from pathlib import Path


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
                calls[call["id"]] = call["function"]
                out["tool_calls"].append({"id": call["id"], "type": "function", "function": {
                    "name": call["function"], "arguments": call["arguments"]}})
        if m["role"] == "tool":
            call_id = m.get("tool_call_id")
            name = m.get("function") or calls.get(call_id)
            if not name:
                raise ValueError(f"Tool message without identifiable function: {call_id}")
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


def export_logs(paths, output, benchmark, provenance):
    from inspect_ai.log import read_eval_log
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, requests, seen, log_results = [], [], set(), []
    for path in paths:
        log = read_eval_log(str(path))
        log_results.append({"path": str(path), "status": log.status,
                            "results": dump(log.results), "stats": dump(log.stats)})
        for sample in log.samples or []:
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
                   "messages": chat, "tools": tools, "scores": scores,
                   "error": dump(sample.error), "limit": dump(sample.limit),
                   "limit_events": [dump(e) for e in sample.events if "limit" in e.event],
                   "generation_token_caps": sum(c.stop_reason == "max_tokens" for e in events for c in e.output.choices),
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
