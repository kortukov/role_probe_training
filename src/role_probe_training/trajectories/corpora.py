"""Lossless adapters for fixed role-training corpora."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def astra_record(row, index, revision):
    source = json.loads(row['trajectory'])
    info = json.loads(row['mcp_info'])['base_info']
    messages, pending = [], []
    for i, original in enumerate(source):
        role = original['role']
        message = {'role': role, 'content': original['content'] or ''}
        if not isinstance(message['content'], str):
            raise ValueError('Unexpected nontext ASTRA content')
        if original.get('reasoning_content'):
            assert role == 'assistant'
            message['thinking'] = original['reasoning_content']
        if original.get('function_call'):
            assert role == 'assistant'
            call = original['function_call']
            call_id = f'astra_{index}_{i}'
            message['tool_calls'] = [{'id': call_id, 'type': 'function', 'function': call}]
            pending.append((call['name'], call_id))
        if role == 'function':
            name = original['name']
            matches = [j for j, (n, _) in enumerate(pending) if n == name]
            if not matches:
                raise ValueError(f'Unmatched ASTRA tool output {index}/{i}/{name}')
            _, call_id = pending.pop(matches[0])
            message.update(role='tool', name=name, tool_call_id=call_id)
        # Speech before a separate tool-call message is commentary, not a turn end.
        if role == 'assistant' and i + 1 < len(source):
            message['channel'] = 'commentary'
            message['ending'] = '<|end|>'
        messages.append(message)
    if pending:
        raise ValueError(f'Unanswered tool calls in ASTRA row {index}')
    return {'schema_version': 1, 'benchmark': 'astra_sft_1k',
            'sample_id': f'astra_{index:04d}', 'trajectory_id': f'astra/{revision}/{index}',
            'source_index': index, 'dataset_split': 'train', 'model': 'upstream_unspecified',
            'messages': messages,
            'tools': [{'type': 'function', 'function': f} for f in info['tool_list']],
            'provenance': {'dataset': 'Emperorizzis/ASTRA-SFT-1k', 'revision': revision,
                           'group_info': info['group_info']},
            'scores': {}, 'error': None, 'export_errors': [], 'limit': None,
            'limit_events': [], 'generation_token_caps': 0}


def depthbench_training(upstream):
    """Reproduce upstream's chain-enabled training selection without its trainer."""
    root = Path(upstream)
    tree = ast.parse((root/'src/tier/data/dataset_loader.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == 'create_seen_unseen_dataset' and len(n.args.args) == 2)
    functions = next(set(ast.literal_eval(n.value.args[0])) for n in method.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == 'training_functions')
    raw = json.loads((root/'data/depthbench/full_dataset.json').read_text())
    no_calls = [k for k, r in raw.items() if r['num_calls'] == 0][:50]
    selected = {k: r for k, r in raw.items() if k in no_calls or
                (r['num_calls'] > 0 and set(r['apis']) <= functions)}
    assert len(selected) == 944
    return selected, functions


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def corpus_summary(rows):
    return {'trajectories': len(rows), 'roles': dict(Counter(
        m['role'] for r in rows for m in r['messages'])),
        'reasoning_bearing': sum(any(m.get('thinking') for m in r['messages']) for r in rows),
        'tool_bearing': sum(any(m['role'] == 'tool' for m in r['messages']) for r in rows),
        'finished_assistant': sum(bool(r['messages']) and r['messages'][-1]['role'] == 'assistant'
                                  and not r['messages'][-1].get('tool_calls') for r in rows)}
