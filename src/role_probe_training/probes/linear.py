"""Multinomial L2 logistic regression, saved as portable NumPy arrays + JSON."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from .harmony import ROLES


@dataclass
class RoleProbe:
    weight: np.ndarray
    bias: np.ndarray
    metadata: dict

    def predict_proba(self, states):
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != self.weight.shape[1]:
            raise ValueError("Expected [content_tokens, hidden_size] pre-MLP states")
        return softmax(states @ self.weight.T + self.bias, axis=-1)

    def predict(self, states):
        roles = np.asarray(self.metadata["roles"])
        return roles[self.predict_proba(states).argmax(axis=-1)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weight=self.weight, bias=self.bias)
        path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            weight, bias = data["weight"], data["bias"]
        metadata = json.loads(path.with_suffix(".json").read_text())
        if weight.shape[0] != len(metadata["roles"]) or bias.shape != (weight.shape[0],):
            raise ValueError("Probe arrays and role metadata disagree")
        return cls(weight, bias, metadata)


def fit_probe(states, labels, *, metadata, C=0.005, max_iter=5000, backend="cuml"):
    """Upstream GPU fitter; CPU backend is for small checks, not this experiment."""
    if set(np.unique(labels)) != set(range(len(ROLES))):
        raise ValueError("All five roles must occur in the training split")
    if backend == "cuml":
        import cupy as cp
        from cuml.linear_model import LogisticRegression as GPULogisticRegression
        model = GPULogisticRegression(penalty="l2", C=C, max_iter=max_iter,
                                      linesearch_max_iter=100, fit_intercept=True,
                                      output_type="numpy")
        x = cp.asarray(states, dtype=cp.float32)
        y = cp.asarray(labels, dtype=cp.int32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, y)
        reference = np.asarray(model.predict_proba(x[:32]))
        weight, bias = np.asarray(model.coef_), np.asarray(model.intercept_)
        iterations = int(np.asarray(model.n_iter_).max())
        warning_messages = [str(w.message) for w in caught]
        del x, y
        cp.get_default_memory_pool().free_all_blocks()
    elif backend == "sklearn":
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=1e-4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(states, labels)
        reference = model.predict_proba(states[:32])
        weight, bias = model.coef_, model.intercept_
        iterations = int(model.n_iter_.max())
        warning_messages = [str(w.message) for w in caught]
    else:
        raise ValueError(f"Unknown probe fitting backend: {backend}")
    probe = RoleProbe(weight, bias, {
        **metadata, "format_version": 1, "roles": list(ROLES),
        "C": C, "scaling": "none", "solver": backend,
        "iterations": iterations, "max_iter": max_iter,
        "warnings": warning_messages,
        "converged": (bool(iterations < max_iter and not warning_messages)
                      if backend == "sklearn" else None),
        "termination_note": "cuML C++ line-search warnings require log review; iteration count alone does not establish convergence.",
    })
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Non-finite fitted probe")
    restored = probe.predict_proba(states[:32].astype(np.float32))
    error = float(np.max(np.abs(restored-reference)))
    probe.metadata['max_saved_probability_error'] = error
    np.testing.assert_allclose(restored, reference, atol=1e-4, rtol=1e-4)
    return probe
