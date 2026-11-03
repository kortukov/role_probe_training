"""Task schedule and provenance checks for fresh, never outcome-filtered data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def schedule(source, validation, seed=123):
    with Path(source).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    counts = Counter(str(row['sample_id']) for row in rows)
    with Path(validation).open() as stream:
        heldout = {str(json.loads(line)['sample_id']) for line in stream if line.strip()}
    if len(rows) != 2700 or len(counts) != 90 or set(counts.values()) != {30}:
        raise ValueError('Require exactly 30 attempts on each of 90 AppWorld training tasks')
    if set(counts) & heldout:
        raise ValueError('Training and validation tasks overlap')
    episodes = [dict(id=f'episode{i:04d}', task_id=str(row['sample_id']),
                     seed=seed + 10000 * i, source_id=row.get('trajectory_id', str(i)))
                for i, row in enumerate(rows)]
    random.Random(seed).shuffle(episodes)
    return [episodes[i:i+16] for i in range(0, len(episodes), 16)]


def validate_requests(requests, episodes, policy_step):
    expected = {x['id'] for x in episodes}
    observed = {x['episode_id'] for x in requests}
    if expected != observed:
        raise ValueError('Generation records do not cover exactly the assigned episodes')
    for request in requests:
        if request['policy_step'] != policy_step:
            raise ValueError('Stale generation policy')
        if not request['prompt_ids'] or not request['completion_ids']:
            raise ValueError('Missing exact generation tokens')
    for episode in expected:
        indices = [x['generation_index'] for x in requests if x['episode_id'] == episode]
        if indices != list(range(len(indices))):
            raise ValueError('Missing, duplicate or reordered generation records')
