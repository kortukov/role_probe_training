"""Deterministic task subsets with configurable effective episode batches."""
from collections import Counter
from role_probe_training.on_policy.data import schedule as original_schedule


def schedule(source, validation, seed=123, *, episodes_per_task=3, batch_size=16):
    if not 1 <= episodes_per_task <= 30 or batch_size < 2 or batch_size % 2:
        raise ValueError('Require 1..30 attempts/task and an even two-rank batch')
    original = [e for batch in original_schedule(source, validation, seed) for e in batch]
    counts = Counter()
    episodes = []
    for episode in original:
        task = episode['task_id']
        if counts[task] < episodes_per_task:
            episodes.append(episode)
            counts[task] += 1
    if len(counts) != 90 or set(counts.values()) != {episodes_per_task}:
        raise ValueError('Incorrect task coverage')
    return [episodes[i:i+batch_size] for i in range(0, len(episodes), batch_size)]
