import json
from pathlib import Path
import subprocess
from role_probe_training.training.data import read_jsonl, trajectory_examples
from role_probe_training.on_policy.data import validate_requests
from .serving import serve

def collect(runtime, config, episodes, step, root, rank, *, sampler=None):
    folder = root/'rollouts'/f'step{step:05d}'/f'rank{rank}'
    if folder.exists():
        raise FileExistsError(f'{folder}: retained attempt needs audit before any retry')
    folder.mkdir(parents=True)
    plan = dict(episodes=episodes, policy_step=step-1, generation=config['generation'],
                image=config['image'], data=config['appworld_data'])
    (folder/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    replay = {}
    recovery = root/'recovery.json'
    if recovery.exists():
        spec = json.loads(recovery.read_text())
        if spec['step'] == step:
            source = Path(spec['source'])/f'rank{rank}'
            if json.loads((source/'plan.json').read_text()) != plan:
                raise ValueError('Recovery batch plan differs')
            from role_probe_training.on_policy.data import sha256
            path = source/'requests.jsonl'
            if sha256(path) != spec['sha256'][str(rank)]:
                raise ValueError('Recovery completion records changed')
            rows = read_jsonl(path)
            replay = {(row['episode_id'],row['generation_index']):row for row in rows}
            if len(replay) != len(rows):
                raise ValueError('Duplicate recovery completions')
    with serve(runtime, folder, step-1, config['generation'], sampler=sampler, replay=replay) as (url, state):
        with (folder/'collector.log').open('w') as stream:
            subprocess.run(['.tools/bin/uv', 'run', '--project', 'src/experiments/agentic_eval/swe',
                '--frozen', '--no-sync', 'python', '-m', 'role_probe_training.on_policy.collect',
                '--plan', str(folder/'plan.json'), '--base-url', url],
                stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=18000)
        if state['errors']:
            raise RuntimeError('Serving errors; inspect retained server_errors.json')
    requests = read_jsonl(folder/'requests.jsonl')
    validate_requests(requests, episodes, step-1)
    records = read_jsonl(folder/'trajectories.jsonl')
    if {x['trajectory_id'] for x in records} != {x['id'] for x in episodes}:
        raise ValueError('Incomplete fresh episode batch')
    examples = trajectory_examples(runtime.tokenizer, folder/'trajectories.jsonl')
    for example in examples:
        if len(example['ids']) > runtime.base.config.max_position_embeddings:
            raise ValueError('Full role history exceeds native limit; retained, not truncated')
    return examples, requests, records

