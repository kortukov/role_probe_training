"""CPU-only check that saved chats work with the pinned HF template.

Counts describe retrospective HF rendering, not the exact native vLLM prompts.
No truncation is applied, and no model weights are loaded.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

p = argparse.ArgumentParser(__doc__)
p.add_argument("input", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model-path", type=Path, default=Path(".cache/models/gpt-oss-20b"))
a = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
rows = []
reader = gzip.open if a.input.suffix == ".gz" else open
with reader(a.input, "rt") as stream:
    for line in stream:
        row = json.loads(line)
        if row["export_errors"] or not row["messages"]:
            raise ValueError(f"Invalid exported chat: {row['sample_id']}")
        ids = tokenizer.apply_chat_template(
            row["messages"], tools=row["tools"],
            reasoning_effort=row["provenance"]["settings"]["reasoning_effort"],
            tokenize=True, add_generation_prompt=False,
        )
        rows.append({"sample_id": row["sample_id"], "tokens": len(ids),
                     "messages": len(row["messages"]),
                     "tool_replies": sum(m["role"] == "tool" for m in row["messages"]),
                     "reasoning_messages": sum(bool(m.get("thinking")) for m in row["messages"])})
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps({
    "rendered_at": datetime.now(timezone.utc).isoformat(),
    "rendering": "HF chat template, retrospective (old reasoning may be dropped)",
    "template_sha256": hashlib.sha256((a.model_path/"chat_template.jinja").read_bytes()).hexdigest(),
    "samples": rows,
}, indent=2) + "\n")
print(f"Tokenized {len(rows)} chats; token lengths {min(r['tokens'] for r in rows)}–{max(r['tokens'] for r in rows)}")
