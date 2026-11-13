"""Run a plan's episodes, concurrently, in the Inspect environment against the live policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import traceback
import urllib.request

from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model
from role_probe_training.trajectories.appworld import appworld_task
from role_probe_training.trajectories.inspect_export import attach_history, export_logs


def run_episode(plan, root, base_url, episode):
    """Return the Inspect log statuses; the exported attempt stays on disk."""
    output = Path(root) / episode['id']
    output.mkdir()
    request = urllib.request.Request(base_url+'/episode',
        data=json.dumps({'id': episode['id'], 'seed': episode['seed']}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert json.load(response)['ok']
    model = get_model('vllm/on-policy-role-model', base_url=f"{base_url}/episodes/{episode['id']}/v1",
        api_key='local', stream=False,
        config=GenerateConfig(temperature=1, top_p=1, seed=episode['seed'],
            max_tokens=plan['generation']['max_tokens'], reasoning_effort='high',
            max_retries=0, timeout=1200, max_connections=1))
    task = appworld_task([episode['task_id']], 'train', plan['image'], plan['data'],
        output/'worlds', episode['id'], max_steps=plan['generation']['max_steps'])
    logs = eval(task, model=model, max_samples=1, max_connections=1,
        time_limit=1800, token_limit=2000000, fail_on_error=False,
        log_dir=str(output/'logs'), log_model_api=True, display='plain')
    export_logs([log.location for log in logs], output, 'appworld_train',
                {'policy_step': plan['policy_step'], 'episode': episode})
    return [log.status for log in logs]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    cached = {r['trajectory_id']: r for r in read_jsonl(root/'cached_trajectories.jsonl')}
    for episode in plan['episodes']:
        if episode['id'] in cached and str(cached[episode['id']]['sample_id']) != episode['task_id']:
            raise ValueError('Recovered episode task mismatch')
    fresh = [episode for episode in plan['episodes'] if episode['id'] not in cached]
    statuses, failures = {}, []
    # One fresh process per episode: Inspect's vLLM provider keeps the first
    # base_url per model name for the life of a process.
    pool = ProcessPoolExecutor(max_workers=args.concurrency, mp_context=multiprocessing.get_context('spawn'),
                               max_tasks_per_child=1)
    with pool:
        futures = {pool.submit(run_episode, plan, root, args.base_url, episode): episode['id']
                   for episode in fresh}
        for future in as_completed(futures):
            try:
                statuses[futures[future]] = future.result()
            except Exception:
                failures.append(f'Episode {futures[future]}:\n{traceback.format_exc()}')
                # Start no new episodes; running ones finish and are retained.
                pool.shutdown(wait=False, cancel_futures=True)
    # Every request has been answered, so the shared server logs are complete.
    events = read_jsonl(root/'parser_events.jsonl')
    stops = read_jsonl(root/'episode_stops.jsonl')
    requests = read_jsonl(root/'requests.jsonl')
    records = []
    for episode in plan['episodes']:
        if episode['id'] in cached:
            row = cached[episode['id']]
        elif episode['id'] in statuses:
            rows = read_jsonl(root/episode['id']/'trajectories.jsonl')
            if len(rows) != 1 or str(rows[0]['sample_id']) != episode['task_id']:
                failures.append(f"Episode {episode['id']}: unexpected task coverage")
                continue
            row = rows[0]
            row['trajectory_id'] = episode['id']
            # The probes read the exact tokens of the episode's last model call.
            attach_history(row, [x for x in requests if x['episode_id'] == episode['id']],
                           sum(x['episode_id'] == episode['id'] for x in stops))
            row['invalid_model_outputs'] = [x for x in events if x['episode_id'] == episode['id'] and x['invalid']]
            row['episode_stops'] = [x for x in stops if x['episode_id'] == episode['id']]
            if row['error'] or row['export_errors'] or row['agent_model_errors']:
                failures.append(f"Episode {episode['id']} failed; retained for audit, no automatic reroll")
            if any(status != 'success' for status in statuses[episode['id']]):
                failures.append(f"Episode {episode['id']}: incomplete Inspect collection")
        else:
            continue
        records.append(row)
    # Persist every attempt, in plan order, before raising; never reroll bad outcomes.
    with (root/'trajectories.jsonl').open('a') as stream:
        for row in records:
            stream.write(json.dumps(row)+'\n')
    if failures:
        raise RuntimeError('\n'.join(failures))
    (root/'collection_complete.json').write_text(json.dumps({
        'policy_step': plan['policy_step'], 'episodes': [x['id'] for x in plan['episodes']],
        'attempts': len(records), 'concurrency': args.concurrency}, indent=2)+'\n')


if __name__ == '__main__':
    main()
