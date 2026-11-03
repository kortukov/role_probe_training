"""Replay saved batch-64 histories through the real tokenizer, without a GPU."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from role_probe_training.on_policy_repaired.serving import template_messages, mixed_channel_template, parse_completion

tokenizer = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b', local_files_only=True)
template = mixed_channel_template(tokenizer.chat_template)
root = Path('experiment_reports/training_against_role_probes/on_policy/appworld270_batch64/runs/epoch1/rollouts/step00001')
checked = mixed = 0
for rank in (0, 1):
    folder = root / f'rank{rank}'
    cache = {}
    for line in (folder/'requests.jsonl').read_text().splitlines():
        row = json.loads(line)
        raw = tokenizer.decode(row['completion_ids'], skip_special_tokens=False)
        parsed = parse_completion(raw)
        if parsed.get('tool_calls'):
            key = tuple(f"{row['episode_id']}_{row['generation_index']+1}_{i}" for i in range(len(parsed['tool_calls'])))
            cache[key] = 'assistant' + raw
    for line in (folder/'trajectories.jsonl').read_text().splitlines():
        record = json.loads(line)
        for end in range(1, len(record['messages'])+1):
            messages = template_messages(record['messages'][:end], cache)
            rendered = tokenizer.apply_chat_template(messages, tools=record['tools'], chat_template=template, tokenize=False, add_generation_prompt=True)
            if any('raw_harmony' in m for m in messages):
                mixed += 1
                for message in messages:
                    if 'raw_harmony' in message:
                        assert '<|start|>'+message['raw_harmony'] in rendered
            else:
                original = tokenizer.apply_chat_template(messages, tools=record['tools'], tokenize=False, add_generation_prompt=True)
                assert rendered == original
            checked += 1
assert mixed > 0
print(json.dumps(dict(prefixes_checked=checked, mixed_prefixes=mixed, unchanged_ordinary_prompts=True, exact_mixed_raw_preserved=True)))
