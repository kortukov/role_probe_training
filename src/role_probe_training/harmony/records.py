"""Token records: the exact prompt and completion of one model call, with message spans.

A record is what the probes consume. Nothing in it is re-rendered later: `prompt_ids`
were the model's input, `completion_ids` its output, and the message entries say which
tokens belong to which message and role.
"""
import numpy as np

from .render import ROLES, Rendering, RenderConfig, completion_prompt, tokens


def request_record(prompt, completion_ids, config, **extra):
    """Record one call. `prompt` is the Rendering of the conversation before the
    `<|start|>assistant` header; completion entries are parsed from the tokens."""
    from .parse import walk_completion
    entries, error = walk_completion(list(completion_ids))
    record = {"prompt_ids": completion_prompt(prompt), "completion_ids": list(completion_ids),
              "prompt_messages": prompt.messages, "completion_messages": entries,
              "completion_parse_error": error, "rendering": config.metadata()}
    record.update(extra)
    return record


def history(record):
    """The whole model-visible history of a call: prompt then completion, as tokens."""
    RenderConfig.from_metadata(record["rendering"])
    ids = list(record["prompt_ids"]) + list(record["completion_ids"])
    rendering = Rendering(ids, [dict(entry) for entry in record["prompt_messages"]])
    shift = len(record["prompt_ids"]) - len(tokens().assistant_header)
    for entry in record["completion_messages"]:
        shifted = dict(entry)
        for key in ("start", "content_start", "content_end", "end"):
            shifted[key] += shift
        shifted["source"] = record.get("completion_source", -1)
        rendering.messages.append(shifted)
    return rendering


def encode_record(record, roles=ROLES):
    """Scoring input for one trajectory row or request record, labelled in one role set.

    Trajectory rows carry their final request under `history`. Rows without
    generation tokens cannot be scored: regenerate them through the Harmony server.
    """
    source = record.get("history", record)
    if "prompt_ids" not in source or "completion_ids" not in source:
        raise ValueError("Record has no generation tokens; regenerate it through the Harmony server "
                         "instead of re-rendering its messages")
    rendering = history(source)
    text, offsets = rendering.text_and_offsets()
    return {"input_ids": np.asarray(rendering.ids, dtype=np.int64), "labels": rendering.labels_for(roles),
            "score_mask": rendering.score_mask(roles), "message_ids": rendering.message_ids, "messages": rendering.messages,
            "source_message_indices": rendering.source_message_indices,
            "text": text, "offsets": offsets, "rendering": source["rendering"]}


def unlabelled_content(rendering):
    """Count content tokens of generated messages whose role is undefined."""
    return int(sum(entry["content_end"] - entry["content_start"] for entry in rendering.messages
                   if entry["kind"] == "assistant" and entry["label"] is None))
