"""Run in the existing Inspect environment against the live training model."""
import argparse
import json
from pathlib import Path
import urllib.request

from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model
from role_probe_training.trajectories.appworld import appworld_task
from role_probe_training.trajectories.inspect_export import export_logs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    records = []
    for episode in plan['episodes']:
        output = root / episode['id']
        output.mkdir()
        request = urllib.request.Request(args.base_url+'/episode',
            data=json.dumps({'id': episode['id']}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            assert json.load(response)['ok']
        model = get_model('vllm/on-policy-role-model', base_url=args.base_url+'/v1', api_key='local', stream=False,
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
        rows = [json.loads(line) for line in (output/'trajectories.jsonl').read_text().splitlines()]
        if len(rows) != 1 or str(rows[0]['sample_id']) != episode['task_id']:
            raise ValueError('Unexpected task coverage')
        row = rows[0]
        row['trajectory_id'] = episode['id']
        records.append(row)
        # Persist every attempt before checking errors, never reroll bad outcomes.
        with (root/'trajectories.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if row['error'] or row['export_errors'] or row['agent_model_errors']:
            raise RuntimeError(f"Episode {episode['id']} failed; retained for audit, no automatic reroll")
        if any(log.status != 'success' for log in logs):
            raise RuntimeError('Incomplete Inspect collection')
    (root/'collection_complete.json').write_text(json.dumps({
        'policy_step': plan['policy_step'], 'episodes': [x['id'] for x in plan['episodes']],
        'attempts': len(records)}, indent=2)+'\n')


if __name__ == '__main__':
    main()
