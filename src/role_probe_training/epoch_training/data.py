"""Matched task visits and fixed-pool selection for rollout refresh experiments."""
import json
from pathlib import Path
import random


def schedule(tasks_path, validation, seed=123, *, epochs=10, attempts_per_task=1, batch_size=16):
    """Episodes of every epoch, in update batches: `attempts_per_task` fresh attempts on every
    training task per epoch, shuffled within the epoch. `tasks_path` lists one task ID per line
    (the AppWorld training split); tasks of the validation rollouts must not be among them."""
    if epochs < 1 or attempts_per_task < 1:
        raise ValueError('epochs and attempts_per_task must be positive')
    tasks = sorted(line.strip() for line in Path(tasks_path).read_text().splitlines() if line.strip())
    with Path(validation).open() as stream:
        heldout = {str(json.loads(line)['sample_id']) for line in stream if line.strip()}
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) & heldout:
        raise ValueError('Training tasks are missing, repeated or overlap the validation tasks')
    batches = []
    for epoch in range(epochs):
        episodes = [dict(id=f'epoch{epoch+1:03d}_task{i:03d}_attempt{attempt:02d}', task_id=task,
                         seed=seed+10000*((epoch*attempts_per_task+attempt)*len(tasks)+i))
                    for attempt in range(attempts_per_task) for i, task in enumerate(tasks)]
        random.Random(seed+epoch).shuffle(episodes)
        batches.extend(episodes[i:i+batch_size] for i in range(0, len(episodes), batch_size))
    return batches


def select_pool(pool, episodes):
    ids = {e['id'] for e in episodes}
    examples = [x for x in pool['examples'] if x['id'] in ids]
    requests = [x for x in pool['requests'] if x['episode_id'] in ids]
    records = [x for x in pool['records'] if x['trajectory_id'] in ids]
    if any(len(xs) != len(ids) for xs in (examples, records)):
        raise ValueError('Epoch pool does not contain exactly one history per assigned episode')
    if {x['episode_id'] for x in requests} != ids:
        raise ValueError('Missing generated requests in epoch pool')
    if any(x['policy_step'] != pool['policy_step'] for x in requests):
        raise ValueError('Mixed policy versions in epoch pool')
    return examples, requests, records
