"""Aggregate split-separated AppWorld scores and rollout timings, without rewriting prose."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text())
    rows=[json.loads(l) for l in (root/'trajectories.jsonl').open()]
    expected=set(manifest['sample_ids'])
    if len(rows)!=len(expected) or {r['sample_id'] for r in rows}!=expected:
        raise ValueError(f'Incomplete or duplicate selection: {root}')
    if any('attachment://' in json.dumps(r['messages']) for r in rows):
        raise ValueError('Unresolved conversation attachments')
    groups=defaultdict(list)
    successes=0
    for r in rows:
        success=not r['error'] and r['scores'].get('appworld_success',{}).get('value') in ('C',1)
        successes+=success
        groups[r['sample_id'].rsplit('_',1)[0]].append(success)
    timing=json.loads((root/'timing.json').read_text())
    durations=[r['rollout_seconds'] for r in timing['samples'] if 'rollout_seconds' in r]
    complete_groups=all(len(v)==3 for v in groups.values())
    return {'source':str(root),'tasks':len(rows),'successes':successes,
        'task_success':successes/len(rows),'errors':sum(bool(r['error']) for r in rows),
        'scenarios':len(groups),'scenario_success':sum(all(v) for v in groups.values())/len(groups) if complete_groups else None,
        'wall_seconds':timing['wall_seconds'],'trajectories_per_hour':timing['trajectories_per_hour'],
        'rollout_median_seconds':statistics.median(durations) if durations else None,
        'concurrency':timing['concurrency']}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',action='append',required=True,help='NAME=RUN_DIRECTORY')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    metrics={name:summarize(Path(path)) for name,path in (x.split('=',1) for x in a.dataset)}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'metrics').mkdir(exist_ok=True)
    (a.output/'metrics/results.json').write_text(json.dumps(metrics,indent=2)+'\n')
    lines=['# Supporting results','',
        'Task success counts all assigned attempts, including errors. Scenario success requires all three task variants to succeed and is omitted for incomplete scenario subsets.', '',
        '| Dataset | Tasks | Successes | Errors | Task success | Scenario success |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        scenario=f"{100*m['scenario_success']:.1f}%" if m['scenario_success'] is not None else '—'
        lines.append(f"| {name} | {m['tasks']} | {m['successes']} | {m['errors']} | {100*m['task_success']:.1f}% | {scenario} |")
    lines+=['','## Collection throughput','',
        'Wall time includes environment initialization, model calls, tool execution and scoring, but excludes inference-server startup. Throughput counts attempts, not successes. Median rollout time excludes environment initialization and is available only for attempts reaching scoring.', '',
        '| Dataset | Concurrent worlds allowed | Collection minutes | Attempts/hour | Median rollout seconds |',
        '|---|---:|---:|---:|---:|']
    for name,m in metrics.items():
        median=f"{m['rollout_median_seconds']:.1f}" if m['rollout_median_seconds'] is not None else '—'
        lines.append(f"| {name} | {m['concurrency']} | {m['wall_seconds']/60:.1f} | {m['trajectories_per_hour']:.1f} | {median} |")
    (a.output/'APPENDIX.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
