"""Independent full-training-set gradient check for exported linear probes."""

import numpy as np
from scipy.special import logsumexp, softmax

from .harmony import ROLES, tokenize_messages
from .linear import RoleProbe
from .model import extract_states


def audit_objective(tokenizer, model, documents, probe_paths, train_sources=None):
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
    # Training-set accuracy, overall and per document collection, from the same logits.
    correct = {layer: {} for layer in probes}
    totals = {}
    role_hits = {layer: np.zeros(len(ROLES)) for layer in probes}
    role_totals = np.zeros(len(ROLES))
    # Full true-role x predicted-role counts, overall and per collection.
    confusion = {layer: {} for layer in probes}
    # Mirror run.py: only the fitted sources' training documents, with their native headers.
    fitted = [d for d in documents if d['split'] == 'train' and (train_sources is None or d['source'] in set(train_sources))]
    for index, document in enumerate(fitted):
        headers = document.get('header', {})
        collection = document.get('collection', 'neutral')
        for label, role in enumerate(ROLES):
            enc = tokenize_messages(tokenizer, [{'role':role, 'content':document['text'], **headers.get(role, {})}])
            positions = np.flatnonzero(enc['labels'] >= 0)
            states = extract_states(model, enc['input_ids'], list(probes), positions)
            count += len(positions)
            totals[collection] = totals.get(collection, 0) + len(positions)
            role_totals[label] += len(positions)
            for layer, probe in probes.items():
                x = states[layer].astype(np.float16).astype(np.float64)
                logits = x @ probe.weight.astype(np.float64).T + probe.bias
                predicted = logits.argmax(axis=1)
                hits = int((predicted == label).sum())
                correct[layer][collection] = correct[layer].get(collection, 0) + hits
                role_hits[layer][label] += hits
                counts = confusion[layer].setdefault(collection, np.zeros((len(ROLES), len(ROLES)), dtype=np.int64))
                counts[label] += np.bincount(predicted, minlength=len(ROLES))
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
            'train_accuracy': sum(correct[layer].values()) / count,
            'train_accuracy_by_collection': {c: correct[layer][c] / totals[c] for c in totals},
            'train_per_role_recall': {ROLES[i]: float(role_hits[layer][i] / role_totals[i]) for i in range(len(ROLES))},
            'train_confusion_counts': sum(confusion[layer].values()).tolist(),
            'train_confusion_counts_by_collection': {c: m.tolist() for c, m in confusion[layer].items()},
            'note': 'Independent float64 gradient of mean cross-entropy plus sample-normalized L2; not an assertion about cuML termination status.',
        }
    return result
