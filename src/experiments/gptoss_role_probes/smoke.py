"""GPU smoke test: loading, content masks and pre-MLP indexing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from role_probe_training.probes.harmony import ROLES, tokenize_messages
from role_probe_training.probes.model import extract_states, load_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='.cache/models/gpt-oss-20b')
args = parser.parse_args()
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
print('GPU:', torch.cuda.get_device_name(), flush=True)
tokenizer, model = load_model(args.model)
for role in ROLES:
    enc = tokenize_messages(tokenizer, [{'role': role, 'content': 'The garden contains green plants and a small wooden bench.'}])
    assert np.all(enc['labels'][enc['labels'] >= 0] == ROLES.index(role))
    positions = np.flatnonzero(enc['labels'] >= 0)
    states = extract_states(model, enc['input_ids'], [12, 16], positions)
    ids = torch.as_tensor(enc['input_ids'], device=model.device).unsqueeze(0)
    reference, handles = {}, []
    for layer in states:
        def before_mlp(module, inputs, layer=layer):
            reference[layer] = inputs[0][0, positions].float().cpu().numpy()
        handles.append(model.model.layers[layer].mlp.register_forward_pre_hook(before_mlp))
    try:
        with torch.inference_mode():
            model.model(ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    for layer in states:
        np.testing.assert_allclose(states[layer], reference[layer], atol=1e-5)
        assert np.isfinite(states[layer]).all()
print('PASS: five role masks, finite states, and norm-output/MLP-input equality at the upstream activation site', flush=True)

from role_probe_training.probes.linear import fit_probe
rng = np.random.default_rng(1)
y = np.repeat(np.arange(5), 30)
x = (np.eye(5)[y]*5 + rng.normal(size=(len(y), 5))*.1).astype(np.float32)
probe = fit_probe(x, y, metadata={'layer':12}, C=1.)
assert (probe.predict(x) == np.asarray(ROLES)[y]).all()
print('PASS: cuML fit and NumPy probability reconstruction', probe.metadata, flush=True)
