"""Token-weighted objectives; KL direction and masking follow upstream honesty."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .data import collate


def kl_sum(runtime, examples, chunk_size=64):
    """KL(base || adapted), at unshifted completion positions, summed over tokens.

    Equivalent to upstream compute_kl_loss's numerator. Checkpoint the vocabulary
    projection/softmax chunks so full-vocabulary distributions are not retained
    for every token during backward. The base uses the same frozen weights with
    adapters disabled; no second model is allocated.
    """
    ids,attention,_,mask=collate(examples,runtime.device,runtime.pad_id)
    with runtime.model.disable_adapter(),torch.no_grad():
        teacher=runtime.base.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
    runtime.hidden=None
    student=runtime.base.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
    runtime.hidden=None
    def chunk_loss(current, reference, score_mask):
        with torch.no_grad():
            base_lp=F.log_softmax(runtime.base.lm_head(reference).float(),dim=-1)
        adapted_lp=F.log_softmax(runtime.base.lm_head(current).float(),dim=-1)
        per_token=F.kl_div(adapted_lp,base_lp.exp(),reduction='none').sum(-1)
        return (per_token*score_mask).sum()
    total=student.sum()*0
    for lo in range(0,ids.shape[1],chunk_size):
        hi=lo+chunk_size
        if not mask[:,lo:hi].any(): continue
        total=total+checkpoint(chunk_loss,student[:,lo:hi],teacher[:,lo:hi],
            mask[:,lo:hi],use_reentrant=False)
    return total
