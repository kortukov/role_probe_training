"""Fit saved standard-data activations in the project's CUDA-13 environment."""

import argparse
import json
from pathlib import Path

import numpy as np

from role_probe_training.probes.evaluation import metrics
from role_probe_training.probes.linear import fit_probe


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    cache = root / 'data/post_training/probe_features'
    states = np.load(cache / 'states.npy', mmap_mode='r')
    labels = np.load(cache / 'labels.npy', mmap_mode='r')
    train = np.load(cache / 'train.npy')
    metadata = json.loads((cache / 'metadata.json').read_text())
    probe = fit_probe(states[train], labels[train], C=0.005, max_iter=5000,
                      backend='cuml', metadata=metadata)
    output = root / 'post_training'
    probe.save(output / 'probes/layer_12.npz')
    result = metrics(labels[~train], probe.predict_proba(states[~train]))
    (output / 'probe_fit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'heldout_standard_accuracy': result['accuracy']}), flush=True)


if __name__ == '__main__':
    main()
