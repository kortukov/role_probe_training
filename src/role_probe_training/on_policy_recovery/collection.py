import json
from pathlib import Path
import subprocess
import sys
from role_probe_training.training.data import read_jsonl, trajectory_examples
from role_probe_training.on_policy.data import validate_requests
from .recovery import stage_cache
from .serving import serve


def collect(runtime, config, episodes, step, root, rank, *, sampler=None):
    concurrency = config.get('collection_concurrency', 1)
    folder = root/'rollouts'/f'step{step:05d}'/f'rank{rank}'
    if folder.exists():
        raise FileExistsError(f'{folder}: retained attempt needs audit before any retry')
    folder.mkdir(parents=True)
    plan = dict(episodes=episodes, policy_step=step-1, generation=config['generation'],
                image=config['image'], data=config['appworld_data'])
    (folder/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    recovery = root/'recovery.json'
    if recovery.exists():
        spec = json.loads(recovery.read_text())
        if spec['step'] == step:
            stage_cache(spec, rank, plan, folder)
    with serve(runtime, folder, step-1, config['generation'], sampler=sampler) as (url, state):
        with (folder/'collector.log').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'role_probe_training.on_policy_recovery.collect',
                '--plan', str(folder/'plan.json'), '--base-url', url, '--concurrency', str(concurrency)],
                stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=config.get('collection_timeout', 18000))
        if state['errors']:
            raise RuntimeError('Serving errors; inspect retained server_errors.json')
    # Concurrent requests interleave; restore the sequential plan/generation order.
    order = {episode['id']: index for index, episode in enumerate(episodes)}
    requests = sorted(read_jsonl(folder/'requests.jsonl'), key=lambda x: (order[x['episode_id']], x['generation_index']))
    validate_requests(requests, episodes, step-1)
    records = read_jsonl(folder/'trajectories.jsonl')
    if {x['trajectory_id'] for x in records} != {x['id'] for x in episodes}:
        raise ValueError('Incomplete fresh episode batch')
    examples = trajectory_examples(folder/'trajectories.jsonl')
    for example in examples:
        if len(example['ids']) > runtime.base.config.max_position_embeddings:
            raise ValueError('Full role history exceeds native limit; retained, not truncated')
    return examples, requests, records
