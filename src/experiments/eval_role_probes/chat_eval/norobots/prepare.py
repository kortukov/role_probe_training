"""Prepare the No Robots test split for two probe evaluations.

Version 1 replays each conversation as given: the sample's own system message (when
present) in the system turn, then the human user/assistant turns, assistant turns as
final-channel answers. Version 2 keeps only the first user turn behind the GPT-OSS
system turn at medium reasoning (plus the sample's system text) so the model can
generate its own reasoning and answer.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET, REVISION = 'HuggingFaceH4/no_robots', 'e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiment_reports/eval_role_probes/chat_eval/norobots/inputs'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, split='test', revision=REVISION, token=False)
    rows, counts = [], Counter()
    for row in dataset:
        messages = [{'role': m['role'], 'content': m['content']} for m in row['messages']]
        has_system = messages[0]['role'] == 'system'
        body = messages[1:] if has_system else messages
        expected = ['user' if i % 2 == 0 else 'assistant' for i in range(len(body))]
        if [m['role'] for m in body] != expected or body[-1]['role'] != 'assistant':
            raise ValueError(f'Unexpected turn structure in {row["prompt_id"]}')
        # Renderer turns: the sample's system message becomes the Harmony developer message; the
        # Harmony system message itself is added at rendering time. Dataset answers are final messages.
        developer = [{'role': 'developer', 'content': messages[0]['content']}] if has_system else []
        replay = developer + [{'role': m['role'], 'content': m['content'],
                               **({'channel': 'final'} if m['role'] == 'assistant' else {})} for m in body]
        prompt = developer + [{'role': 'user', 'content': body[0]['content']}]
        rows.append({'id': row['prompt_id'], 'category': row['category'], 'has_system': has_system,
                     'n_user_turns': len(body) // 2, 'replay_turns': replay, 'prompt_turns': prompt})
        counts['with_system'] += has_system
        counts['multi_turn'] += len(body) > 2
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (args.output / 'samples.jsonl').write_text(text)
    manifest = {'dataset': DATASET, 'revision': REVISION, 'split': 'test', 'n_samples': len(rows),
                'with_system_message': counts['with_system'], 'multi_turn': counts['multi_turn'],
                'user_turns': dict(Counter(r['n_user_turns'] for r in rows)),
                'categories': dict(Counter(r['category'] for r in rows)),
                'version2_system_turn': 'Harmony system message (reasoning medium, fixed date) rendered by oss-harmony; the sample system message, when present, is the developer message',
                'samples_sha256': hashlib.sha256(text.encode()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'categories'}))


if __name__ == '__main__':
    main()
