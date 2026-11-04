"""Matched task visits and fixed-pool selection for rollout refresh experiments."""
import random

from role_probe_training.on_policy_recovery.data import schedule as task_schedule


def schedule(source, validation, seed=123, *, epochs=10, batch_size=16):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    tasks = sorted(e['task_id'] for batch in task_schedule(
        source, validation, seed, episodes_per_task=1, batch_size=batch_size) for e in batch)
    batches = []
    for epoch in range(epochs):
        episodes = [dict(id=f'epoch{epoch+1:03d}_task{i:03d}', task_id=task,
                         seed=seed+10000*(epoch*len(tasks)+i)) for i, task in enumerate(tasks)]
        random.Random(seed+epoch).shuffle(episodes)
        batches.extend(episodes[i:i+batch_size] for i in range(0, len(episodes), batch_size))
    return batches


def select_pool(pool, episodes):
    ids = {e['id'] for e in episodes}
    examples = [x for x in pool['examples'] if x['id'] in ids]
    requests = [x for x in pool['requests'] if x['episode_id'] in ids]
    records = [x for x in pool['records'] if x['trajectory_id'] in ids]
    if any(len(xs) != len(ids) for xs in (examples, records)):
        raise ValueError('Epoch pool does not contain exactly one history per assigned task')
    if {x['episode_id'] for x in requests} != ids:
        raise ValueError('Missing generated requests in epoch pool')
    if any(x['policy_step'] != pool['policy_step'] for x in requests):
        raise ValueError('Mixed policy versions in epoch pool')
    return examples, requests, records
