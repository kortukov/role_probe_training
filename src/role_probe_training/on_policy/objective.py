"""Exact conditional reverse KL on the contexts used to sample each action."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def reverse_kl_sum(runtime, request, chunk_size=64):
    """Sum KL(current || base) before each sampled token, including controls.

    The complete generation context is replayed once. Vocabulary projections are
    checkpointed in chunks; attention histories are never chunked. Histories are
    fixed for this gradient: this is not a trajectory policy-gradient estimator.
    """
    prompt = request['prompt_ids']
    completion = request['completion_ids']
    if not prompt or not completion:
        raise ValueError('KL requires a nonempty prompt and sampled completion')
    tokens = prompt + completion
    if len(tokens) > runtime.base.config.max_position_embeddings:
        raise ValueError('Generation context exceeds native limit; no truncation')
    ids = torch.tensor([tokens], device=runtime.device)
    attention = torch.ones_like(ids)
    with runtime.model.disable_adapter(), torch.no_grad():
        reference = runtime.base.model(input_ids=ids, attention_mask=attention,
                                       use_cache=False).last_hidden_state
    runtime.hidden = None
    current = runtime.base.model(input_ids=ids, attention_mask=attention,
                                 use_cache=False).last_hidden_state
    runtime.hidden = None

    def projected_kl(q, p):
        with torch.no_grad():
            log_p = F.log_softmax(runtime.base.lm_head(p).float(), dim=-1)
        log_q = F.log_softmax(runtime.base.lm_head(q).float(), dim=-1)
        return (log_q.exp() * (log_q - log_p)).sum()

    total = current.sum() * 0
    # Hidden state t predicts token t+1, including the FIRST completion token.
    start, stop = len(prompt) - 1, len(tokens) - 1
    for lo in range(start, stop, chunk_size):
        hi = min(lo + chunk_size, stop)
        total = total + checkpoint(projected_kl, current[:, lo:hi],
                                   reference[:, lo:hi], use_reentrant=False)
    return total
