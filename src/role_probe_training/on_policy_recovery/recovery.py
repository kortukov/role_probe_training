"""Recover completed same-policy data, never re-execute tools or replay prompts."""
import json
from pathlib import Path
from role_probe_training.on_policy.data import sha256, validate_requests


def stage_cache(spec, rank, plan, folder):
    source=Path(spec['cache'])/f'rank{rank}'
    manifest=spec['ranks'][str(rank)]
    for name,digest in manifest['sha256'].items():
        if sha256(source/name)!=digest:
            raise ValueError(f'Recovery cache changed: {source/name}')
    old_plan=json.loads((source/'plan.json').read_text())
    if old_plan!=plan:
        raise ValueError('Recovery batch plan differs')
    rows=[json.loads(line) for line in (source/'trajectories.jsonl').read_text().splitlines()]
    requests=[json.loads(line) for line in (source/'requests.jsonl').read_text().splitlines()]
    ids={r['trajectory_id'] for r in rows}
    if len(ids)!=len(rows):raise ValueError('Duplicate recovered episode')
    assigned={e['id']:e for e in plan['episodes']}
    if not ids<=assigned.keys():raise ValueError('Recovered episode not in batch')
    for row in rows:
        if str(row['sample_id'])!=assigned[row['trajectory_id']]['task_id']:
            raise ValueError('Recovered task mismatch')
        if row['provenance']['policy_step']!=plan['policy_step']:
            raise ValueError('Recovered policy mismatch')
        if row['error'] and not row.get('recovered_terminal'):
            raise ValueError('Unclassified failed episode cannot be reused')
    validate_requests(requests,[e for e in plan['episodes'] if e['id'] in ids],plan['policy_step'])
    (folder/'cached_trajectories.jsonl').write_text((source/'trajectories.jsonl').read_text())
    (folder/'requests.jsonl').write_text((source/'requests.jsonl').read_text())
    (folder/'recovery_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
