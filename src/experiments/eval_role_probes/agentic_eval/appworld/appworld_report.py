"""Aggregate disjoint AppWorld shards and timings without rewriting authored prose."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics


def summarize(roots):
    rows, samples, timings, selections = [], [], [], []
    for root in roots:
        manifest = json.loads((root / 'manifest.json').read_text())
        part = [json.loads(line) for line in (root / 'trajectories.jsonl').open()]
        expected = set(manifest['sample_ids'])
        if len(part) != len(expected) or {r['sample_id'] for r in part} != expected:
            raise ValueError(f'Incomplete or duplicate shard: {root}')
        selections.append(set(manifest.get('full_selection', manifest['sample_ids'])))
        rows.extend(part)
        timing = json.loads((root / 'timing.json').read_text())
        timings.append(timing)
        samples.extend(timing['samples'])
    if any(s != selections[0] for s in selections):
        raise ValueError('Shards have different full selections')
    if len(rows) != len(selections[0]) or {r['sample_id'] for r in rows} != selections[0]:
        raise ValueError('Missing or duplicated tasks across shards')
    if any('attachment://' in json.dumps(r['messages']) for r in rows):
        raise ValueError('Unresolved conversation attachments')
    groups = defaultdict(list)
    successes = 0
    for row in rows:
        success = not row['error'] and row['scores'].get('appworld_success', {}).get('value') in ('C', 1)
        successes += success
        groups[row['sample_id'].rsplit('_', 1)[0]].append(success)
    durations = [sample['rollout_seconds'] for sample in samples if 'rollout_seconds' in sample]
    if len(roots) == 1:
        elapsed = timings[0]['wall_seconds']
    else:
        starts = [datetime.fromisoformat(t['started_utc']) for t in timings]
        ends = [datetime.fromisoformat(t['completed_utc']) for t in timings]
        elapsed = (max(ends) - min(starts)).total_seconds()
    complete_groups = all(len(values) == 3 for values in groups.values())
    metrics = {
        'sources': [str(root) for root in roots], 'tasks': len(rows), 'successes': successes,
        'task_success': successes / len(rows), 'errors': sum(bool(r['error']) for r in rows),
        'reasoning_bearing': sum(any(m.get('thinking') for m in r['messages']) for r in rows),
        'tool_bearing': sum(any(m['role'] == 'tool' for m in r['messages']) for r in rows),
        'step_limited': sum(s.get('step_limit', False) for s in samples),
        'scenarios': len(groups),
        'scenario_success': sum(all(v) for v in groups.values()) / len(groups) if complete_groups else None,
        'wall_seconds': elapsed, 'trajectories_per_hour': len(rows) * 3600 / elapsed,
        'rollout_median_seconds': statistics.median(durations) if durations else None,
        'gpus': len(roots), 'concurrency': sum(t['concurrency'] for t in timings),
    }
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', action='append', required=True, help='NAME=RUN_DIRECTORY[,RUN_DIRECTORY]')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (args.output / 'metrics').mkdir(parents=True, exist_ok=True)
    (args.output / 'data').mkdir(exist_ok=True)
    metrics = {}
    for spec in args.dataset:
        name, paths = spec.split('=', 1)
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Dataset names must be simple file-safe identifiers')
        metrics[name], rows = summarize([Path(path) for path in paths.split(',')])
        # Timing pilots repeat training tasks; keep them out of the canonical corpus.
        if not name.startswith('pilot_'):
            target = args.output / 'data' / f'{name}.jsonl'
            temporary = target.with_suffix('.tmp')
            with temporary.open('w') as stream:
                for row in sorted(rows, key=lambda r: r['sample_id']):
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            temporary.replace(target)
    (args.output / 'metrics/results.json').write_text(json.dumps(metrics, indent=2) + '\n')
    lines = ['# Supporting results', '',
        'Task success counts all assigned attempts, including errors. Scenario success requires all three task variants to succeed and is omitted for incomplete scenario subsets. Step-limited attempts exhausted the Python interaction allowance.', '',
        '| Dataset | Tasks | Successes | Errors | Step-limited | Task success | Scenario success |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, result in metrics.items():
        scenario = f"{100 * result['scenario_success']:.1f}%" if result['scenario_success'] is not None else '—'
        lines.append(f"| {name} | {result['tasks']} | {result['successes']} | {result['errors']} | {result['step_limited']} | {100 * result['task_success']:.1f}% | {scenario} |")
    lines += ['', '## Collection throughput', '',
        'Collection time includes environment initialization, model calls, tool execution and scoring, but excludes inference-server startup and export. Parallel shards use the interval from the earliest start to the latest finish on the same node. Throughput counts attempts, not successes. Median rollout time excludes initialization and is available only for attempts reaching scoring.', '',
        '| Dataset | GPUs | Concurrent worlds allowed | Collection minutes | Attempts/hour | Median rollout seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for name, result in metrics.items():
        median = f"{result['rollout_median_seconds']:.1f}" if result['rollout_median_seconds'] is not None else '—'
        lines.append(f"| {name} | {result['gpus']} | {result['concurrency']} | {result['wall_seconds'] / 60:.1f} | {result['trajectories_per_hour']:.1f} | {median} |")
    lines += ['', '## Trajectory coverage', '',
        'Reasoning-bearing trajectories contain an explicit assistant thinking field. Tool-bearing trajectories contain at least one tool reply. Attempts lacking either are retained, not silently removed.', '',
        '| Dataset | With reasoning | With tool replies | Total |',
        '|---|---:|---:|---:|']
    for name, result in metrics.items():
        lines.append(f"| {name} | {result['reasoning_bearing']} | {result['tool_bearing']} | {result['tasks']} |")
    (args.output / 'APPENDIX.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
