"""Independent full-training-set gradient check for exported linear probes."""

import numpy as np
from scipy.special import logsumexp, softmax

from .harmony import ROLES, tokenize_messages
from .linear import RoleProbe
from .model import extract_states


def audit_objective(tokenizer, model, documents, probe_paths):
    """Stream all fit tokens; accumulate gradients in float64 on the CPU.

    Objective: mean multinomial cross-entropy + ||W||²/(2*C*n).
    This checks stationarity independently of cuML's stopping messages. A small
    gradient is diagnostic, not a guarantee about out-of-distribution accuracy.
    """
    probes = {layer: RoleProbe.load(path) for layer, path in probe_paths.items()}
    gw = {layer: np.zeros_like(p.weight, dtype=np.float64) for layer,p in probes.items()}
    gb = {layer: np.zeros_like(p.bias, dtype=np.float64) for layer,p in probes.items()}
    loss = {layer: 0.0 for layer in probes}
    count = 0
    for index, document in enumerate(d for d in documents if d['split'] == 'train'):
        for label, role in enumerate(ROLES):
            enc = tokenize_messages(tokenizer, [{'role':role, 'content':document['text']}])
            positions = np.flatnonzero(enc['labels'] >= 0)
            states = extract_states(model, enc['input_ids'], list(probes), positions)
            count += len(positions)
            for layer, probe in probes.items():
                x = states[layer].astype(np.float16).astype(np.float64)
                logits = x @ probe.weight.astype(np.float64).T + probe.bias
                loss[layer] += float((logsumexp(logits,axis=1)-logits[:,label]).sum())
                residual = softmax(logits,axis=1)
                residual[:,label] -= 1
                gw[layer] += residual.T @ x
                gb[layer] += residual.sum(axis=0)
        if (index+1) % 50 == 0:
            print(f'Audited {index+1} fitting documents', flush=True)
    result = {}
    for layer, probe in probes.items():
        weight = probe.weight.astype(np.float64)
        weight_gradient = gw[layer]/count + weight/(probe.metadata['C']*count)
        bias_gradient = gb[layer]/count
        maximum = float(max(np.abs(weight_gradient).max(), np.abs(bias_gradient).max()))
        result[str(layer)] = {
            'n_train_tokens': count,
            'mean_cross_entropy': loss[layer]/count,
            'objective': loss[layer]/count + float((weight**2).sum())/(2*probe.metadata['C']*count),
            'max_abs_gradient': maximum,
            'gradient_below_1e_4': maximum < 1e-4,
            'note': 'Independent float64 gradient of mean cross-entropy plus sample-normalized L2; not an assertion about cuML termination status.',
        }
    return result
