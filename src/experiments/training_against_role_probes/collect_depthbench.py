"""Collect independently seeded GPT-OSS histories in local DepthBench environments."""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from openai import AsyncOpenAI
from role_probe_training.trajectories.depthbench import DepthBenchEnvironment, tool_schemas

SYSTEM = ('You are a helpful assistant with access to tools. Use the available tools when needed '
          'to answer the user accurately. Base factual claims on returned data. If tools cannot '
          'fulfill the request, explain the limitation. Complete the task and give a concise final answer.')


async def collect(a):
    root = a.root
    tasks = [json.loads(line) for n in range(7) for line in (root/f'data/tasks/calls_{n}.jsonl').open()]
    if a.smoke:
        tasks = [next(r for r in tasks if r['num_calls'] == n) for n in range(7)]
    manifest = json.loads((root/'data_manifest.json').read_text())
    schemas = tool_schemas(root/'data/upstream')
    by_name = {x['function']['name']: x for x in schemas}
    base_env = DepthBenchEnvironment(root/'data/upstream')
    client = AsyncOpenAI(base_url=a.base_url, api_key='local', timeout=300, max_retries=2)
    semaphore = asyncio.Semaphore(a.concurrency)
    a.output.mkdir(parents=True, exist_ok=True)
    settings = {'dataset_revision': manifest['revision'], 'attempts': a.attempts,
        'temperature': 1, 'reasoning_effort': 'medium', 'max_tokens_per_call': 8192,
        'max_assistant_turns': 24, 'extra_functions': 3, 'seed_formula': '9100000 + task_id*10000 + attempt*100 + turn',
        'model': 'openai/gpt-oss-20b', 'model_revision': '6cee5e81ee83917806bbde320786a8fb61efebee',
        'smoke': a.smoke, 'concurrency': a.concurrency,
        'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment_sha256': hashlib.sha256(Path('src/role_probe_training/trajectories/depthbench.py').read_bytes()).hexdigest()}
    mp = a.output/'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == settings, 'Resume configuration changed'
    mp.write_text(json.dumps(settings, indent=2)+'\n')
    async def rollout(task, attempt):
        identifier = f'depthbench/{task["task_id"]}/attempt{attempt:02d}'
        target = a.output/f'calls_{task["num_calls"]}'/f'{int(task["task_id"]):04d}_{attempt:02d}.json'
        if target.exists():
            previous = json.loads(target.read_text())
            assert previous['trajectory_id'] == identifier
            return previous
        async with semaphore:
            start = time.monotonic()
            seed = 9100000 + int(task['task_id'])*10000 + attempt*100
            rng = random.Random(123 + int(task['task_id']))
            relevant = set(task['apis'])
            extras = rng.sample(sorted(set(by_name)-relevant), 3)
            names = sorted(relevant | set(extras))
            rng.shuffle(names)
            tools = [by_name[n] for n in names]
            env = copy.deepcopy(base_env)
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task['prompt']}]
            api_messages = copy.deepcopy(messages)
            raw_calls = []
            error, limit = None, None
            for turn in range(24):
                try:
                    response = await client.chat.completions.create(
                        model='openai/gpt-oss-20b', messages=api_messages, tools=tools,
                        temperature=1, seed=seed+turn, reasoning_effort='medium', max_tokens=8192)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    break
                raw = response.model_dump(exclude_none=True)
                raw_calls.append(raw)
                choice = response.choices[0]
                m = choice.message.model_dump(exclude_none=True)
                content = m.get('content') or ''
                thinking = m.get('reasoning_content') or m.get('reasoning') or ''
                if not isinstance(thinking, str):
                    raise ValueError('Nontext reasoning; add a metadata adapter')
                calls = m.get('tool_calls') or []
                normalized = {'role': 'assistant', 'content': content}
                if thinking:
                    normalized['thinking'] = thinking
                if calls:
                    normalized['tool_calls'] = calls
                messages.append(normalized)
                api_message = {'role': 'assistant', 'content': content}
                if thinking:
                    api_message['reasoning_content'] = thinking
                if calls:
                    api_message['tool_calls'] = calls
                api_messages.append(api_message)
                if choice.finish_reason == 'length':
                    limit = 'generation_tokens'
                    break
                if not calls:
                    break
                for call in calls:
                    f = call['function']
                    result = (env.execute(f['name'], f['arguments']) if f['name'] in names else
                              {'status_code': 400, 'error': 'Tool not available in this task'})
                    observation = {'role': 'tool', 'name': f['name'], 'tool_call_id': call['id'],
                                   'content': json.dumps(result, ensure_ascii=False, default=str)}
                    messages.append(observation)
                    api_messages.append(observation)
            else:
                limit = 'assistant_turns'
            row = {'schema_version': 1, 'benchmark': 'depthbench_train', 'sample_id': identifier,
                'trajectory_id': identifier, 'task_id': task['task_id'], 'attempt': attempt,
                'required_calls': task['num_calls'], 'generation_seed': seed, 'dataset_split': 'train',
                'model': settings['model'], 'messages': messages, 'tools': tools,
                'provenance': {'dataset_revision': manifest['revision'], 'settings': settings},
                'scores': {}, 'error': error, 'limit': limit, 'limit_events': [],
                'generation_token_caps': int(limit == 'generation_tokens'), 'export_errors': [],
                'seconds': time.monotonic()-start, 'generation_records': len(raw_calls)}
            target.parent.mkdir(parents=True, exist_ok=True)
            raw_path = target.with_suffix('.generations.json')
            raw_path.write_text(json.dumps(raw_calls, ensure_ascii=False)+'\n')
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(row, ensure_ascii=False)+'\n')
            tmp.replace(target)
            print(f'{identifier} calls={task["num_calls"]} turns={len(raw_calls)} error={bool(error)} limit={limit} {row["seconds"]:.1f}s', flush=True)
            return row
    # Interleave task groups, keeping all 20 attempts identifiable and resumable.
    work = [(t, attempt) for attempt in range(a.attempts) for t in tasks]
    rows = await asyncio.gather(*(rollout(t, i) for t, i in work))
    summary = {'trajectories': len(rows), 'errors': sum(bool(r['error']) for r in rows),
        'limits': dict(Counter(r['limit'] for r in rows)),
        'reasoning_bearing': sum(any(m.get('thinking') for m in r['messages']) for r in rows),
        'by_required_calls': dict(Counter(r['required_calls'] for r in rows))}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    await client.close()
    if summary['errors']:
        raise RuntimeError(f'Collection has {summary["errors"]} model API errors; audit before assembly')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('experiment_reports/training_against_role_probes/depthbench'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--attempts', type=int, default=20)
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--smoke', action='store_true')
    asyncio.run(collect(p.parse_args()))


if __name__ == '__main__':
    main()
