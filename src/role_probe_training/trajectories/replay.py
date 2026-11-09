"""Scoring input for recorded trajectories: the exact tokens the model saw and wrote.

`encode_record` reads the token record the Harmony server attached to a trajectory
(`history`). Message text is never re-rendered; see role_probe_training.harmony.
"""
from role_probe_training.harmony.records import encode_record  # noqa: F401
