"""The probe fitter of the training_against_probes paper, with our five-class token loss.

It lives in training/ because every training manifest hashes this folder; the from-scratch
`adamw` backend of probes/linear.py uses the same code.

This follows `train_combined_probes` in submodules/training_against_probes/src/probes/train.py
(revision 9c3f80c), which upstream uses for the step-0 fit before model training, for every
refit inside the training loop and for fresh probes on trained checkpoints:

* the probe reads activations divided by one fixed scale, sqrt(mean ||x||^2) over the fitting
  tokens (upstream `normalize_input: unit_norm`); warm-started refits keep the step-0 scale;
* a new AdamW optimizer per fit: learning rate 1e-3, weight decay 1e-4, betas (0.9, 0.95);
* the transformers cosine schedule with 50 warmup steps over `steps` scheduler steps;
* batches of 32 examples (here passage-role examples with every content token), reshuffled
  each pass; gradient clipping at norm 1; the forward pass under bfloat16 autocast;
* upstream's number of optimizer steps (`optimizer_steps`): normally `steps + 1`, because it
  leaves its loop only after the batch whose zero-based index equals `max_steps`; the first
  and the last of them then have learning rate 0.

The only intended difference is the loss. Upstream pools a sequence to one binary logit; here
every content token is one five-class example and a batch's loss is the mean cross-entropy over
its tokens.
"""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.95)
WARMUP_STEPS = 50
CLIP_GRAD_NORM = 1.0
BATCH_EXAMPLES = 32
STEP0_STEPS = 5000  # upstream probe_training.max_steps
SETTINGS = {
    "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "betas": list(BETAS),
    "schedule": "cosine", "warmup_steps": WARMUP_STEPS, "clip_grad_norm": CLIP_GRAD_NORM,
    "batch_examples": BATCH_EXAMPLES, "autocast": "bfloat16", "input_normalization": "unit_norm",
    "loss": "mean five-class cross-entropy over the content tokens of a batch",
    "reference": "training_against_probes 9c3f80c, src/probes/train.py::train_combined_probes",
}


class ScaledLinear(torch.nn.Linear):
    """A linear probe applied to activations divided by a fixed scale.

    `raw()` returns the equivalent weights for unscaled activations, which is how probes are
    stored (RoleProbe). `rescale` changes the scale without changing the function.
    """

    def __init__(self, in_features, out_features, *, input_scale=None, device=None, dtype=None):
        super().__init__(in_features, out_features, device=device, dtype=dtype)
        self.register_buffer("input_scale", torch.tensor(float(input_scale or 1.), device=device, dtype=torch.float32))
        self.register_buffer("scale_set", torch.tensor(input_scale is not None, device=device))

    def forward(self, x):
        return F.linear(x / self.input_scale, self.weight, self.bias)

    @torch.no_grad()
    def rescale(self, scale):
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid probe input scale {scale}")
        self.weight.mul_(scale / float(self.input_scale))
        self.input_scale.fill_(scale)
        self.scale_set.fill_(True)

    @torch.no_grad()
    def load_raw(self, weight, bias, input_scale=None):
        """Load weights stored for unscaled activations."""
        self.weight.copy_(torch.as_tensor(weight, device=self.weight.device))
        self.bias.copy_(torch.as_tensor(bias, device=self.bias.device))
        self.input_scale.fill_(1.)
        self.scale_set.fill_(False)
        if input_scale is not None:
            self.rescale(input_scale)

    @torch.no_grad()
    def raw(self):
        weight = (self.weight.double() / self.input_scale.double()).float()
        return weight.cpu().numpy(), self.bias.detach().float().cpu().numpy()


def input_scale(sum_squares, tokens):
    """sqrt(mean ||x||^2), upstream `compute_input_scales`."""
    return math.sqrt(float(sum_squares) / float(tokens))


def optimizer_steps(steps, examples, batch_size=BATCH_EXAMPLES):
    """Optimizer steps of upstream's loop for max_steps=`steps` over `examples` examples.

    It schedules ceil(steps * min(batch, examples) / examples) passes and breaks after the
    batch whose zero-based index equals `steps`: steps + 1 batches, unless the passes end first
    (only when they hold exactly `steps` batches).
    """
    passes = math.ceil(steps * min(batch_size, examples) / examples)
    return min(steps + 1, passes * math.ceil(examples / batch_size))


def fit(probe, batch, examples, counts, *, steps, batch_size=BATCH_EXAMPLES, seed=123, sync=None):
    """Run one upstream fit on `probe` in place; returns the learning rate of every optimizer step.

    `batch(indices)` returns this process's (features, labels) for those examples, or None if it
    holds none of them. `counts[i]` is the token count of example i over all processes, so the
    loss is the mean over the whole batch. `sync` sums gradients over processes in place.
    """
    parameters = list(probe.parameters())
    examples = list(examples)
    probe.requires_grad_(True)
    probe.train()
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=BETAS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=steps)
    generator = random.Random(seed)
    order, cursor, rates = [], 0, []
    device_type = parameters[0].device.type
    for _ in range(optimizer_steps(steps, len(examples), batch_size)):
        if cursor >= len(order):
            order = list(examples); generator.shuffle(order); cursor = 0
        indices = order[cursor:cursor+batch_size]; cursor += len(indices)
        denominator = sum(counts[i] for i in indices)
        local = batch(indices)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if local is None:
                loss = sum(p.sum()*0 for p in parameters)
            else:
                loss = F.cross_entropy(probe(local[0]), local[1], reduction="sum") / denominator
        loss.backward()
        if sync is not None:
            for parameter in parameters:
                sync(parameter.grad)
        torch.nn.utils.clip_grad_norm_(parameters, CLIP_GRAD_NORM)
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    probe.requires_grad_(False)
    return rates


def fit_from_scratch(states, labels, groups, *, classes, steps=STEP0_STEPS, seed=123, device=None):
    """Upstream step-0 fit: a freshly initialised probe on tokens grouped into examples.

    `groups[i]` names the example (one rendered passage-role message) that token i belongs
    to; an example's tokens must be contiguous. Returns the probe, its input scale and the
    number of optimizer steps taken.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    groups = np.asarray(groups)
    if len(groups) != len(states) or len(labels) != len(states) or not len(states):
        raise ValueError("States, labels and groups must be non-empty and aligned")
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    if len(np.unique(groups)) != len(starts):
        raise ValueError("The tokens of one example must be contiguous")
    ends = np.r_[starts[1:], len(groups)]
    x = torch.as_tensor(np.ascontiguousarray(states), device=device)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    sum_squares = sum(float(x[lo:lo+65536].double().square().sum()) for lo in range(0, len(x), 65536))
    scale = input_scale(sum_squares, len(x))
    # torch.nn.Linear's default initialisation, drawn from a seeded generator.
    probe = ScaledLinear(x.shape[1], classes, input_scale=scale)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(probe.weight, a=math.sqrt(5), generator=generator)
        bound = 1 / math.sqrt(probe.in_features)
        torch.nn.init.uniform_(probe.bias, -bound, bound, generator=generator)
    probe.to(device)

    def batch(indices):
        rows = torch.cat([torch.arange(starts[i], ends[i], device=device) for i in indices])
        return x[rows].float(), y[rows]

    rates = fit(probe, batch, range(len(starts)), (ends-starts).tolist(), steps=steps, seed=seed)
    return probe, scale, len(rates)
