"""Fit saved standard-data activations in the probe-fitting environment
(src/experiments/gptoss_role_probes/fit_env, which holds cuML and torch).

`--backend cuml` (default) writes post_training/probes/layer_12.npz and probe_fit.json.
`--backend adamw` runs the training_against_probes step-0 fit on the same activations and
writes post_training/probes_adamw/layer_12.npz and probe_fit_adamw.json, so both probes can
be compared.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from role_probe_training.probes.evaluation import metrics
from role_probe_training.probes.linear import fit_probe

BACKENDS = ('cuml', 'adamw')


def suffix(backend):
    return '' if backend == 'cuml' else f'_{backend}'


def probe_path(root, backend):
    return root / f'post_training/probes{suffix(backend)}/layer_12.npz'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backend', choices=BACKENDS, default='cuml')
    args = parser.parse_args()
    root, backend = args.root, args.backend
    cache = root / 'data/post_training/probe_features'
    states = np.load(cache / 'states.npy', mmap_mode='r')
    labels = np.load(cache / 'labels.npy', mmap_mode='r')
    train = np.load(cache / 'train.npy')
    metadata = json.loads((cache / 'metadata.json').read_text())
    groups = None
    if backend == 'adamw':
        if not (cache / 'groups.npy').exists():
            raise FileNotFoundError(f'{cache}/groups.npy is missing: this activation cache predates the adamw '
                                    'backend; rerun the fit stage to extract it again')
        groups = np.load(cache / 'groups.npy')[train]
    probe = fit_probe(states[train], labels[train], C=0.005, max_iter=5000,
                      backend=backend, groups=groups, metadata=metadata)
    output = root / 'post_training'
    probe.save(probe_path(root, backend))
    result = metrics(labels[~train], probe.predict_proba(states[~train]))
    (output / f'probe_fit{suffix(backend)}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'backend': backend, 'heldout_standard_accuracy': result['accuracy']}), flush=True)


if __name__ == '__main__':
    main()
