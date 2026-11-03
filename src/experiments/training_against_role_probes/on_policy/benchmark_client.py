"""Matched serving diagnostic; its short task budget is not a training setting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path('experiment_reports/training_against_role_probes/on_policy/appworld2700')
TASKS = ['b7a9ee9_1', '82e2fac_1']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['warmup', 'micro', 'episodes'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'episodes':
        from inspect_ai import eval
        from inspect_ai.model import GenerateConfig, get_model
        from role_probe_training.trajectories.appworld import appworld_task
        from role_probe_training.trajectories.inspect_export import export_logs
        results = []
        for concurrency in (1, 2):
            output = args.output/f'episodes_c{concurrency}'
            output.mkdir()
            model = get_model('vllm/'+args.model, base_url=args.url+'/v1', api_key='local', stream=False,
                config=GenerateConfig(temperature=1, top_p=1, seed=123, max_tokens=8192,
                    reasoning_effort='high', max_retries=0, timeout=600, max_connections=concurrency))
            task = appworld_task(TASKS, 'train', '.cache/appworld/appworld.sif', '.cache/appworld/data',
                output/'worlds', output.name, max_steps=50)
            started = time.perf_counter()
            logs = eval(task, model=model, max_samples=concurrency, max_connections=concurrency,
                time_limit=300, token_limit=2000000, fail_on_error=False,
                log_dir=str(output/'logs'), log_model_api=True, display='plain')
            seconds = time.perf_counter()-started
            export_logs([log.location for log in logs], output, 'appworld_train', {
                'purpose':'serving benchmark only', 'task_time_limit_seconds':300,
                'concurrency':concurrency, 'model':args.model})
            rows = [json.loads(line) for line in (output/'trajectories.jsonl').read_text().splitlines()]
            assert {str(row['sample_id']) for row in rows} == set(TASKS)
            results.append(dict(concurrency=concurrency, wall_seconds=seconds, attempts=len(rows),
                sample_errors=sum(bool(row['error']) for row in rows),
                limited_samples=sum(bool(row['limit']) or bool(row['limit_events']) for row in rows),
                generation_calls=sum(row['generation_records'] for row in rows),
                generated_tokens=sum(v.get('output_tokens',0) for row in rows for v in row['model_usage'].values())))
            (args.output/'episodes.json').write_text(json.dumps(results,indent=2)+'\n')
        return

    sources = [ROOT/'runs/epoch1/rollouts/step00001'/rank/episode/'generations.jsonl'
               for rank,episode in [('rank0','episode1233'),('rank1','episode2121')]]
    payloads = []
    for source in sources:
        body = json.loads(source.read_text().splitlines()[0])['call']['request']
        body.pop('extra_headers', None)
        body.update(model=args.model, max_tokens=16 if args.phase=='warmup' else 128,
                    temperature=1, top_p=1, seed=123)
        payloads.append(body)

    def generate(index):
        body = {**payloads[index%2], 'seed':123+index}
        request = urllib.request.Request(args.url+'/v1/chat/completions', data=json.dumps(body).encode(),
                                        headers={'Content-Type':'application/json'})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        seconds = time.perf_counter()-started
        return dict(index=index, seconds=seconds, prompt_tokens=result['usage']['prompt_tokens'],
                    completion_tokens=result['usage']['completion_tokens'],
                    finish_reason=result['choices'][0]['finish_reason'])

    results = []
    for concurrency in ((1,) if args.phase=='warmup' else (1,4)):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            rows = list(executor.map(generate, range(1 if args.phase=='warmup' else 4)))
        seconds = time.perf_counter()-started
        tokens = sum(row['completion_tokens'] for row in rows)
        results.append(dict(concurrency=concurrency, wall_seconds=seconds, output_tokens=tokens,
                            output_tokens_per_second=tokens/seconds, requests=rows))
        (args.output/f'{args.phase}.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
