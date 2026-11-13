"""Fit fresh probes on saved standard-data activations with the project's fitter (L-BFGS).

Reads <root>/data/post_training/probe_features (labels.npy, train.npy, metadata.json and
states.npy, or one states_<layer>.npy per layer named in metadata['layers']) and writes
<root>/post_training/probes/layer_<layer>.npz and probe_fit.json. probe_fit.json holds the
held-out metrics of the first layer at the top level and those of every layer under `layers`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from role_probe_training.probes.evaluation import metrics
from role_probe_training.probes.linear import fit_probe


def probe_path(root, layer=12):
    return root / f'post_training/probes/layer_{layer}.npz'


def fit_cache(root):
    cache = root / 'data/post_training/probe_features'
    labels = np.load(cache / 'labels.npy', mmap_mode='r')
    train = np.load(cache / 'train.npy')
    metadata = json.loads((cache / 'metadata.json').read_text())
    layers = metadata.get('layers') or [metadata['layer']]
    results = {}
    for layer in layers:
        name = f'states_{layer}.npy' if 'layers' in metadata else 'states.npy'
        states = np.load(cache / name, mmap_mode='r')
        probe = fit_probe(states[train], labels[train], metadata={**metadata, 'layer': layer})
        if not probe.metadata['converged']:
            raise RuntimeError(f'Layer-{layer} probe fit did not converge: {probe.metadata["fit"]}')
        probe.save(probe_path(root, layer))
        results[str(layer)] = metrics(labels[~train], probe.predict_proba(states[~train]))
        print(json.dumps({'layer': layer, 'heldout_standard_accuracy': results[str(layer)]['accuracy'],
                          'iterations': probe.metadata['iterations']}), flush=True)
    (root / 'post_training/probe_fit.json').write_text(
        json.dumps({**results[str(layers[0])], 'layers': results}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    fit_cache(parser.parse_args().root)
