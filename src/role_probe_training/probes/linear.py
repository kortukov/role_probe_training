"""Linear role probes, saved as portable NumPy arrays + JSON.

A probe has one class per role of its role set (`roles` in its metadata): the five roles
or the header roles of role_probe_training.harmony.

Two ways to fit one from scratch (`fit_probe(backend=...)`):

* "cuml": multinomial L2 logistic regression solved on the GPU, the fitter of the role-probe
  paper (submodules/prompt-injection-as-role-confusion). "sklearn" is its CPU stand-in for checks.
* "adamw": the gradient-descent fitter of the training_against_probes paper
  (training/probe_fit.py), which the training loop also uses for its refits.

Either way the stored weights apply to raw activations.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from role_probe_training.harmony import ROLES


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


def mean_norm_scale(states, chunk=65536):
    """sqrt(mean ||x||^2) over tokens: the input scale of the training_against_probes fitter."""
    total = sum(float(np.square(states[lo:lo+chunk], dtype=np.float64).sum()) for lo in range(0, len(states), chunk))
    return float(np.sqrt(total / len(states)))


def fit_probe(states, labels, *, metadata, roles=ROLES, C=0.005, max_iter=5000, backend="cuml", groups=None, seed=123,
              dtype="float32", solver_options=None, on_step=None):
    """Fit a probe from scratch; `labels` index into `roles` (ROLES or HEADER_ROLES).

    backend "cuml" (or "sklearn" for small CPU checks) solves L2 logistic regression with
    strength C in at most max_iter solver iterations. Backend "adamw" runs the
    training_against_probes step-0 fit for max_iter optimizer steps; it ignores C and needs
    `groups`, the example (rendered passage-role message) each token belongs to, because it
    batches by example.

    `dtype` and `solver_options` (extra cuML LogisticRegression arguments such as tol,
    lbfgs_memory or verbose) leave the cuML objective unchanged and only control how
    precisely it is solved. For the adamw backend `solver_options` may hold `learning_rate`, a
    peak rate other than upstream's 1e-3; `on_step` is that fitter's monitoring hook.
    """
    roles = tuple(roles)
    if set(np.unique(labels)) != set(range(len(roles))):
        raise ValueError(f"All {len(roles)} roles must occur in the training split")
    extra = {}
    if backend == "cuml":
        import cupy as cp
        from cuml.linear_model import LogisticRegression as GPULogisticRegression
        model = GPULogisticRegression(penalty="l2", C=C, max_iter=max_iter, fit_intercept=True, output_type="numpy",
                                      **{"linesearch_max_iter": 100, **(solver_options or {})})
        # Cast on the GPU in pieces: a double-precision copy of the activations does not fit in host memory.
        x = cp.empty(states.shape, dtype=dtype)
        for lo in range(0, len(states), 65536):
            x[lo:lo+65536] = cp.asarray(states[lo:lo+65536])
        y = cp.asarray(labels, dtype=cp.int32)
        extra = {"dtype": str(dtype), "solver_options": {"linesearch_max_iter": 100, **(solver_options or {})}}
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
    elif backend == "adamw":
        if groups is None:
            raise ValueError("The adamw backend batches by example and needs `groups`")
        import torch
        from role_probe_training.training.probe_fit import SETTINGS, fit_from_scratch
        options = {"learning_rate": SETTINGS["learning_rate"], **(solver_options or {})}
        if set(options) != {"learning_rate"}:
            raise ValueError("The adamw backend accepts only the solver option learning_rate")
        module, scale, taken = fit_from_scratch(states, labels, groups, classes=len(roles), steps=max_iter, seed=seed,
                                                on_step=on_step, **options)
        with torch.no_grad():
            head = torch.as_tensor(np.asarray(states[:32], dtype=np.float32), device=module.weight.device)
            reference = torch.softmax(module(head).double(), dim=-1).cpu().numpy()
        weight, bias = module.raw()
        iterations, warning_messages, C = max_iter, [], None
        extra = {"input_scale": scale, "fit": {**SETTINGS, **options, "optimizer_steps": taken, "seed": seed,
                                               "examples": int(len(np.unique(groups)))}}
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unknown probe fitting backend: {backend}")
    notes = {
        "cuml": "cuML C++ line-search warnings require log review; iteration count alone does not establish convergence.",
        "sklearn": "scikit-learn L-BFGS; converged means it stopped before max_iter without warnings.",
        "adamw": "Fixed-length AdamW schedule (training_against_probes step-0 fit); it is not run to a convergence criterion.",
    }
    probe = RoleProbe(weight, bias, {
        # input_scale describes the fitting activations; the stored weights apply to raw
        # activations (scaling none). The training loop reuses it for its refits.
        "input_scale": extra["input_scale"] if "input_scale" in extra else mean_norm_scale(states),
        **metadata, "format_version": 1, "roles": list(roles),
        "C": C, "scaling": "none", "solver": backend,
        "iterations": iterations, "max_iter": max_iter,
        "warnings": warning_messages,
        "converged": (bool(iterations < max_iter and not warning_messages)
                      if backend == "sklearn" else None),
        "termination_note": notes[backend], **extra,
    })
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Non-finite fitted probe")
    restored = probe.predict_proba(states[:32].astype(np.float32))
    error = float(np.max(np.abs(restored-reference)))
    probe.metadata['max_saved_probability_error'] = error
    np.testing.assert_allclose(restored, reference, atol=1e-4, rtol=1e-4)
    return probe
