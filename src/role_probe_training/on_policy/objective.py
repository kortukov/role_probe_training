"""Exact conditional reverse KL on the contexts used to sample each action."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def reverse_kl_sum(runtime, request, chunk_size=64):
    """Sum KL(current || base) before each sampled token of one request."""
    return reverse_kl_sums(runtime, [request], chunk_size)


def reverse_kl_sums(runtime, requests, chunk_size=64):
    """Sum KL(current || base) before each sampled token, including controls.

    Each complete generation context is replayed once; several requests share a
    right-padded pass. Attention is causal, so no real position attends to the
    padding after its own sequence. Vocabulary projections are checkpointed in
    chunks; attention histories are never chunked. Histories are fixed for this
    gradient: this is not a trajectory policy-gradient estimator.
    """
    items = []
    for request in requests:
        prompt, completion = request['prompt_ids'], request['completion_ids']
        if not prompt or not completion:
            raise ValueError('KL requires a nonempty prompt and sampled completion')
        # Hidden state t predicts token t+1, including the FIRST completion token.
        items.append(dict(tokens=prompt + completion, columns=list(range(len(prompt) - 1, len(prompt) + len(completion) - 1))))
    return reverse_kl_histories(runtime, items, chunk_size)


def episode_histories(requests):
    """One replay per episode in place of one per model call, with the same KL positions.

    The requests of an episode extend each other: a later prompt starts with the earlier
    prompt and completion (the Harmony server checks this on every call). Under causal
    attention the hidden state before a sampled token is then the same in the earlier
    request and in the episode's last one, so the whole episode needs a single pass over its
    last request with the positions of every call's sampled tokens. Only the last token of
    an earlier completion may differ (a final answer's <|return|> is replayed as <|end|>);
    it lies after every position that predicts that completion. An episode whose requests
    do not extend each other this way keeps one item per request.
    """
    episodes = {}
    for request in requests:
        episodes.setdefault(request['episode_id'], []).append(request)
    items = []
    for group in episodes.values():
        group = sorted(group, key=lambda request: request['generation_index'])
        last = group[-1]
        tokens = last['prompt_ids'] + last['completion_ids']
        columns, consistent = [], True
        for request in group:
            prompt, completion = request['prompt_ids'], request['completion_ids']
            if not prompt or not completion:
                raise ValueError('KL requires a nonempty prompt and sampled completion')
            context = prompt + completion[:-1]  # everything a position predicting this completion can see
            if tokens[:len(context)] != context:
                consistent = False
                break
            columns += range(len(prompt) - 1, len(prompt) + len(completion) - 1)
        if consistent and len(set(columns)) == len(columns):
            items.append(dict(tokens=tokens, columns=columns))
        else:
            items += [dict(tokens=r['prompt_ids'] + r['completion_ids'],
                           columns=list(range(len(r['prompt_ids']) - 1, len(r['prompt_ids']) + len(r['completion_ids']) - 1)))
                      for r in group]
    return items


def reverse_kl_histories(runtime, items, chunk_size=64):
    """Sum KL(current || base) at the given positions of each token sequence.

    `items` are {'tokens', 'columns'}: hidden state `column` predicts the sampled token at
    `column + 1`. Several sequences share a right-padded pass.
    """
    sequences, rows, columns = [], [], []
    for row, item in enumerate(items):
        tokens = item['tokens']
        if len(tokens) > runtime.base.config.max_position_embeddings:
            raise ValueError('Generation context exceeds native limit; no truncation')
        sequences.append(tokens)
        columns += item['columns']
        rows += [row] * len(item['columns'])
    width = max(map(len, sequences))
    ids = torch.tensor([x + [runtime.pad_id] * (width - len(x)) for x in sequences], device=runtime.device)
    attention = torch.ones_like(ids)
    with runtime.model.disable_adapter(), torch.no_grad():
        reference = runtime.base.model(input_ids=ids, attention_mask=attention,
                                       use_cache=False).last_hidden_state
    runtime.hidden = None
    current = runtime.base.model(input_ids=ids, attention_mask=attention,
                                 use_cache=False).last_hidden_state
    runtime.hidden = None
    rows = torch.tensor(rows, device=runtime.device)
    columns = torch.tensor(columns, device=runtime.device)
    current, reference = current[rows, columns], reference[rows, columns]

    def projected_kl(q, p):
        with torch.no_grad():
            log_p = F.log_softmax(runtime.base.lm_head(p).float(), dim=-1)
        log_q = F.log_softmax(runtime.base.lm_head(q).float(), dim=-1)
        return (log_q.exp() * (log_q - log_p)).sum()

    total = current.sum() * 0
    for lo in range(0, len(current), chunk_size):
        total = total + checkpoint(projected_kl, current[lo:lo + chunk_size],
                                   reference[lo:lo + chunk_size], use_reentrant=False)
    return total


def kl_batches(requests, max_tokens=None):
    """Group requests so each padded pass holds at most max_tokens positions.

    None keeps one request per pass. Longest requests come first, so a pass is
    never wider than a request that would run alone.
    """
    if max_tokens is None:
        return [[request] for request in requests]
    def length(request):
        return len(request['prompt_ids']) + len(request['completion_ids'])
    batches = []
    for request in sorted(requests, key=length, reverse=True):
        if batches and (len(batches[-1]) + 1) * length(batches[-1][0]) <= max_tokens:
            batches[-1].append(request)
        else:
            batches.append([request])
    return batches
