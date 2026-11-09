"""Generate GPT-OSS-20B reasoning and answers for No Robots first turns via a local vLLM server.

The prompt is rendered by role_probe_training.harmony (oss-harmony) and sent as token
IDs; vLLM returns completion token IDs. Each record holds the exact prompt and
completion tokens with their message spans, so scoring reads precisely what the model
saw and wrote.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from role_probe_training.harmony import RenderConfig, completion_prompt, encoding, render_turns, request_record, tokens, walk_completion


def read_jsonl(path):
    # Split on newlines only: json.dumps(ensure_ascii=False) leaves U+2028-style separators inside strings.
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n') if line.strip()]


def check_chat_completion(completion):
    """Reject anything but complete analysis/final messages ending with <|return|>."""
    entries, error = walk_completion(completion)
    if error is not None:
        return entries, f'unparseable: {error}'
    if not completion or completion[-1] != tokens().ret or not entries or not entries[-1]['complete']:
        return entries, 'generation did not finish a final answer'
    for entry in entries:
        if entry['role'] != 'assistant' or entry['recipient'] or entry['channel'] not in ('analysis', 'final'):
            return entries, f"unexpected generated message: channel {entry['channel']!r} recipient {entry['recipient']!r}"
    if entries[-1]['channel'] != 'final':
        return entries, 'last message is not final'
    return entries, None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=Path('experiment_reports/eval_role_probes/chat_eval/norobots/inputs/samples.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='openai/gpt-oss-20b')
    parser.add_argument('--max-tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--reasoning-effort', default='medium')
    parser.add_argument('--date', default='2025-12-31', help='Harmony system message date')
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key='local', timeout=900, max_retries=2)
    rows = read_jsonl(args.input)[:args.limit]
    config = RenderConfig(args.reasoning_effort, args.date)

    def generate(row):
        rendering = render_turns(row['prompt_turns'], config)
        prompt = completion_prompt(rendering)
        response = client.completions.create(
            model=args.model, prompt=prompt, max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=1.0, seed=args.seed,
            extra_body={'return_token_ids': True, 'stop_token_ids': list(tokens().stop), 'skip_special_tokens': False})
        choice = response.choices[0]
        extra = choice.model_extra or {}
        completion = extra.get('token_ids')
        if not completion or extra.get('prompt_token_ids') != prompt:
            raise ValueError('Exact generation token IDs missing or changed')
        entries, reason = check_chat_completion(completion)
        record = request_record(rendering, completion, config, id=row['id'], category=row['category'],
                                has_system=row['has_system'], finish_reason=choice.finish_reason,
                                prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                                completion_source=len(row['prompt_turns']), ok=reason is None)
        if reason is None:
            record['messages'] = row['prompt_turns'] + [{'role': 'assistant', 'channel': e['channel'], 'content': e['content']}
                                                       for e in entries]
        else:
            record['reason'] = reason
            record['raw'] = encoding().decode(completion)
        return record

    started = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        records = list(pool.map(generate, rows))
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records if r['ok']))
    failures = [r for r in records if not r['ok']]
    args.output.with_name('generation_failures.json').write_text(json.dumps(failures, indent=1, ensure_ascii=False) + '\n')
    completed = [r for r in records if r['ok']]
    summary = {'n_prompts': len(records), 'completed': len(completed), 'failed': len(failures),
               'failure_reasons': dict(Counter(f"{r['finish_reason']}: {r['reason']}" for r in failures)),
               'mean_completion_tokens': sum(r['completion_tokens'] for r in records) / max(1, len(records)),
               'mean_reasoning_characters_completed': sum(len(m['content']) for r in completed for m in r['messages']
                                                          if m.get('channel') == 'analysis') / max(1, len(completed)),
               'elapsed_seconds': elapsed, 'settings': {k: v for k, v in vars(args).items() if k not in ('input', 'output')},
               'rendering': config.metadata()}
    args.output.with_name('generation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'settings'}))


if __name__ == '__main__':
    main()
