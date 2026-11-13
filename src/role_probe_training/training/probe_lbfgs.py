"""The one probe fitter: full-batch L-BFGS (torch.optim.LBFGS) run to a gradient test.

It fits every probe: the from-scratch probes of `probes/linear.py` (`fit_probe`), the
fresh evaluation probes of trained checkpoints and the warm-started refits
inside the training loop (`refit.py`). It lives in training/ because every training manifest
hashes this folder.

Objective (the role-probe paper's L2 logistic regression, as its cuML solver defines it):

    mean cross-entropy over all fitting tokens + ||W||^2 / (2 C n)

with n the number of fitting tokens, raw (unscaled) activations and an unpenalised bias.
It is strictly convex in W, so a warm start changes the time to the minimum, not the minimum.

Raw activations make this objective badly conditioned (feature means far from zero, strongly
correlated features on very different scales), and L-BFGS on W itself stalls just short of
the gradient test. The solve therefore runs in whitened coordinates, V = W @ A and
c = b + W @ mean with A = (feature covariance + I / (C n))^(1/2), which is the same objective
and the same minimum written in other variables. The returned weights and the convergence
test are in the original coordinates.

The solve is in double precision. Loss and gradient are accumulated over chunks of tokens
and, with `sync`, summed over processes, so the fitting activations need not fit on one GPU.
A fit is converged when the largest absolute gradient entry of the objective is at most
`gradient_tolerance`; reaching `max_iterations` first is reported as not converged and is
never treated as success.
"""
import json
import math
import time

import torch
import torch.nn.functional as F

C = 0.005  # the role-probe paper's L2 strength
GRADIENT_TOLERANCE = 1e-6
MAX_ITERATIONS = 5000
SETTINGS = {
    "optimizer": "torch.optim.LBFGS", "line_search": "strong_wolfe", "history_size": 30, "precision": "float64",
    "batch": "all fitting tokens", "input_normalization": "none",
    "solve_coordinates": "V = W @ (feature covariance + I/(C n))^(1/2), c = b + W @ feature mean (same objective and minimum)",
    "objective": "mean cross-entropy + ||W||^2 / (2 C n), bias unpenalised",
    "convergence_test": "largest absolute gradient entry <= gradient_tolerance",
}


def fit(features, labels, *, classes, weight=None, bias=None, C=C, tokens=None, sync=None,
        gradient_tolerance=GRADIENT_TOLERANCE, max_iterations=MAX_ITERATIONS, block_iterations=25,
        chunk_tokens=65536, log=None):
    """Minimise the objective; returns (weight, bias, info) as float32 tensors for raw activations.

    `features` [tokens, hidden] and `labels` are this process's fitting tokens on the solve
    device, in any float dtype (they are cast to double chunk by chunk). `weight` and `bias`
    warm-start the solve; without them it starts from zero. With `sync` (an in-place sum over
    processes such as `torch.distributed.all_reduce`) every process must call `fit` with the
    same start and pass the global token count as `tokens`; all then follow the same iterates.
    `log(dict)` receives a progress record every ten blocks of `block_iterations` iterations.
    """
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Expected [tokens, hidden] features aligned with labels")
    if C <= 0 or gradient_tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid L2 strength, gradient tolerance or iteration limit")
    device = features.device
    n = int(len(labels) if tokens is None else tokens)
    if n < 1:
        raise ValueError("Empty fitting split")
    strength = 1. / (C * n)
    # Feature mean and covariance over all fitting tokens (summed over processes).
    hidden = features.shape[1]
    total = torch.zeros(hidden, device=device, dtype=torch.float64)
    moments = torch.zeros((hidden, hidden), device=device, dtype=torch.float64)
    for lo in range(0, len(labels), chunk_tokens):
        chunk = features[lo:lo+chunk_tokens].double()
        total += chunk.sum(0); moments += chunk.T @ chunk
    if sync is not None:
        sync(total); sync(moments)
    mean = total / n
    spectrum, basis = torch.linalg.eigh(moments / n - torch.outer(mean, mean))
    spectrum = (spectrum.clamp_min(0) + strength).sqrt()
    whiten, unwhiten = (basis / spectrum) @ basis.T, (basis * spectrum) @ basis.T  # A^-1 and A
    del moments, basis
    # w and b are the probe on raw activations; v and c are the variables L-BFGS moves.
    w = torch.zeros((classes, features.shape[1]), device=device, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(classes, device=device, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        if weight is not None:
            w.copy_(torch.as_tensor(weight, device=device))
            b.copy_(torch.as_tensor(bias, device=device))
    v = (w.detach() @ unwhiten).requires_grad_()
    c = (b.detach() + w.detach() @ mean).requires_grad_()
    last, evaluations, started = {}, 0, time.perf_counter()

    def closure():
        nonlocal evaluations
        with torch.no_grad():
            w.copy_(v @ whiten); b.copy_(c - w @ mean)
        w.grad, b.grad = torch.zeros_like(w), torch.zeros_like(b)
        cross_entropy = torch.zeros((), device=device, dtype=torch.float64)
        for lo in range(0, len(labels), chunk_tokens):
            logits = F.linear(features[lo:lo+chunk_tokens].double(), w, b)
            loss = F.cross_entropy(logits, labels[lo:lo+chunk_tokens], reduction="sum") / n
            loss.backward()
            cross_entropy += loss.detach()
        if sync is not None:  # processes without tokens take part in the same sums
            for value in (w.grad, b.grad, cross_entropy):
                sync(value)
        with torch.no_grad():
            penalty = .5 * strength * w.square().sum()
            w.grad.add_(w, alpha=strength)
            gradient_max = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
            v.grad = (w.grad - torch.outer(b.grad, mean)) @ whiten  # chain rule to the solve variables
            c.grad = b.grad.clone()
        value = cross_entropy + penalty
        last.update(objective=float(value), cross_entropy=float(cross_entropy), gradient_max=gradient_max)
        evaluations += 1
        if not math.isfinite(last["objective"]) or not math.isfinite(gradient_max):
            raise RuntimeError("Non-finite probe objective or gradient")
        return value

    optimizer = torch.optim.LBFGS([v, c], lr=1., max_iter=block_iterations, max_eval=2*block_iterations,
                                  tolerance_grad=0., tolerance_change=0.,  # the test below, in probe coordinates, decides
                                  history_size=SETTINGS["history_size"], line_search_fn=SETTINGS["line_search"])
    closure()
    initial = dict(last)
    iterations = lambda: optimizer.state[v].get("n_iter", 0)
    for block in range(math.ceil(max_iterations / block_iterations)):
        if last["gradient_max"] <= gradient_tolerance or iterations() >= max_iterations:
            break
        optimizer.param_groups[0]["max_iter"] = min(block_iterations, max_iterations - iterations())
        optimizer.step(closure)
        closure()
        if log is not None and block % 10 == 9:
            log(dict(event="probe_lbfgs_progress", iterations=iterations(), **last))
    solution = dict(last)
    # The stored probe is float32: report the objective and gradient at the stored weights too.
    closure()  # w and b at the final v and c
    stored_w, stored_b = w.detach().float(), b.detach().float()
    with torch.no_grad():
        v.copy_(stored_w.double() @ unwhiten); c.copy_(stored_b.double() + stored_w.double() @ mean)
    closure()
    info = dict(
        solver="lbfgs", settings=SETTINGS, C=C, training_tokens=n, gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations, converged=solution["gradient_max"] <= gradient_tolerance,
        iterations=iterations(), function_evaluations=evaluations, warm_start=weight is not None,
        initial_objective=initial["objective"], initial_gradient_max=initial["gradient_max"], **solution,
        stored_objective=last["objective"], stored_gradient_max=last["gradient_max"],
        fit_seconds=time.perf_counter() - started)
    return stored_w, stored_b, info


def print_log(record):
    print(json.dumps(record), flush=True)
