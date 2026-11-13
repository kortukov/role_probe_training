"""Linear role probes, saved as portable NumPy arrays + JSON.

A probe has one class per role of its role set (`roles` in its metadata): the five roles
or the header roles of role_probe_training.harmony.

Every probe is fitted by the project's one fitter (training/probe_lbfgs.py): full-batch
torch.optim.LBFGS run to a gradient test on the role-probe paper's L2 logistic-regression
objective. The training loop uses the same fitter, warm-started, for its refits. The stored
weights apply to raw activations; prediction needs only NumPy/SciPy.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from role_probe_training.harmony import ROLES
from role_probe_training.training import probe_lbfgs


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

    @property
    def roles(self):
        return tuple(self.metadata["roles"])

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


def fit_probe(states, labels, *, metadata, roles=ROLES, C=probe_lbfgs.C, max_iter=probe_lbfgs.MAX_ITERATIONS,
              gradient_tolerance=probe_lbfgs.GRADIENT_TOLERANCE):
    """Fit a probe from scratch; `labels` index into `roles` (ROLES or HEADER_ROLES).

    Solves L2 logistic regression with strength C in at most max_iter L-BFGS iterations, on
    the GPU when there is one. The metadata records whether the gradient test was met
    (`converged`); callers decide what an unconverged fit means for them.
    """
    roles = tuple(roles)
    if set(np.unique(labels)) != set(range(len(roles))):
        raise ValueError(f"All {len(roles)} roles must occur in the training split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move in pieces and keep the stored dtype: the solver casts to double chunk by chunk.
    x = torch.empty(states.shape, dtype=torch.from_numpy(states[:1].copy()).dtype, device=device)
    for lo in range(0, len(states), 65536):
        x[lo:lo+65536] = torch.from_numpy(np.ascontiguousarray(states[lo:lo+65536]))
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    w, b, info = probe_lbfgs.fit(x, y, classes=len(roles), C=C, max_iterations=max_iter,
                                 gradient_tolerance=gradient_tolerance, log=probe_lbfgs.print_log)
    with torch.no_grad():
        reference = torch.softmax(x[:32].double() @ w.double().T + b.double(), dim=-1).cpu().numpy()
    weight, bias = w.cpu().numpy(), b.cpu().numpy()
    del x, y, w, b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    probe = RoleProbe(weight, bias, {
        **metadata, "format_version": 1, "roles": list(roles),
        "C": C, "scaling": "none", "solver": "lbfgs",
        "iterations": info["iterations"], "max_iter": max_iter,
        "warnings": [] if info["converged"] else [
            f"L-BFGS stopped at the iteration limit with largest gradient entry {info['gradient_max']:.3g}"],
        "converged": info["converged"],
        "termination_note": "Converged means: the largest absolute gradient entry of the objective is at most "
                            "gradient_tolerance.",
        "fit": info,
    })
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Non-finite fitted probe")
    restored = probe.predict_proba(states[:32].astype(np.float32))
    probe.metadata['max_saved_probability_error'] = float(np.max(np.abs(restored-reference)))
    np.testing.assert_allclose(restored, reference, atol=1e-4, rtol=1e-4)
    return probe
