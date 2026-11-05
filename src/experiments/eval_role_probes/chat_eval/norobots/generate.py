"""Generate GPT-OSS-20B reasoning and answers for No Robots first turns via a local vLLM server.

The prompt is the exact Harmony prefix the probes later score, rendered by
harmony.render and tokenized by the server with the same tokenizer and no added
special tokens, so generation and scoring contexts match.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from role_probe_training.probes.harmony import parse_generated, render

ENDINGS = {200002: '<|return|>', 200012: '<|call|>', 199999: '<|endoftext|>'}


def read_jsonl(path):
    # Split on newlines only: json.dumps(ensure_ascii=False) leaves U+2028-style separators inside strings.
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n') if line.strip()]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=Path('experiment_reports/eval_role_probes/chat_eval/norobots/inputs/samples.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--max-tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key='local', timeout=900, max_retries=2)
    rows = read_jsonl(args.input)[:args.limit]

    def generate(row):
        prefix, _ = render(row['prompt_messages'], bos=True)
        prefix += '<|start|>assistant'
        response = client.completions.create(
            model='openai/gpt-oss-20b', prompt=prefix, max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=1.0, seed=args.seed,
            extra_body={'skip_special_tokens': False, 'spaces_between_special_tokens': False, 'add_special_tokens': False})
        choice = response.choices[0]
        stop_reason = getattr(choice, 'stop_reason', None)
        if stop_reason is None and getattr(choice, 'model_extra', None):
            stop_reason = choice.model_extra.get('stop_reason')
        ending = ENDINGS.get(stop_reason) if isinstance(stop_reason, int) else None
        if choice.finish_reason == 'stop' and ending is None:
            ending = '<|return|>'  # the GPT-OSS EOS that vLLM consumed
        continuation = '<|start|>assistant' + choice.text + (ending or '')
        record = {'id': row['id'], 'category': row['category'], 'has_system': row['has_system'],
                  'finish_reason': choice.finish_reason, 'stop_reason': stop_reason,
                  'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'raw': continuation}
        try:
            record['messages'] = row['prompt_messages'] + parse_generated(continuation)
            record['ok'] = True
        except ValueError as error:
            record['ok'], record['reason'] = False, str(error)
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
               'reasoning_level': 'medium (system prompt)'}
    args.output.with_name('generation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'settings'}))


if __name__ == '__main__':
    main()
