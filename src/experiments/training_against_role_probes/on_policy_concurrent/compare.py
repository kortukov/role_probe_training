"""Compare collection throughput and training metrics of sequential and concurrent runs."""
import argparse
import json
from pathlib import Path


def events(run):
    return [json.loads(line) for line in (Path(run)/'rank0.jsonl').read_text().splitlines()]


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(run):
    steps = {}
    local = {e['step']: e['local_episodes'] for e in events(run) if e['event'] == 'collection_complete'}
    for event in events(run):
        if event['event'] == 'update':
            folder = Path(run)/'rollouts'/f"step{event['step']:05d}"/'rank0'
            cached = {x['trajectory_id'] for x in lines(folder/'cached_trajectories.jsonl')}
            fresh = [x for x in lines(folder/'requests.jsonl') if x['episode_id'] not in cached]
            episodes = local[event['step']] - len(cached)
            steps[event['step']] = dict(
                fresh_episodes=episodes, cached_episodes=len(cached),
                fresh_requests=len(fresh), generated_tokens=sum(len(x['completion_ids']) for x in fresh),
                collection_seconds=round(event['collection_seconds'], 1),
                seconds_per_fresh_episode=round(event['collection_seconds']/episodes, 1) if episodes else None,
                total_seconds=round(event['total_seconds'], 1),
                successes=f"{event['collection_successes']}/{event['collection_attempts']}",
                role_loss=round(event['role_loss'], 4), role_accuracy=round(event['role_accuracy'], 4),
                reverse_kl=round(event['reverse_kl'], 5))
    validation = {e['step']: dict(accuracy=round(e['metrics']['accuracy'], 4),
                                  macro_recall=round(e['metrics']['macro_recall'], 4))
                  for e in events(run) if e['event'] == 'validation'}
    return dict(run=str(run), steps=steps, validation=validation)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sequential', type=Path, required=True)
    parser.add_argument('--concurrent', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = dict(sequential=summarize(args.sequential), concurrent=summarize(args.concurrent))
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
