"""Verify repaired decode masks against explicit causal/window/padding truth."""
import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import masking_utils as masks
from role_probe_training.generation.masks import eager_decode_masks

original=masks.create_block_mask
cases=0
with eager_decode_masks():
    for length in (1,127,128,129,433,434,512,1025):
        for window in (None,128):
            for cropped in (False,True):
                offset=max(0,length-128) if cropped else 0
                kv_length=length-offset
                padding=torch.ones((1,length),dtype=torch.bool)
                padding[0,0]=False
                function=masks.causal_mask_function if window is None else masks.sliding_window_causal_mask_function(window)
                block=masks.flex_attention_mask(batch_size=1,cache_position=torch.tensor([length-1]),
                    kv_length=kv_length,kv_offset=offset,mask_function=function,attention_mask=padding)
                actual=create_mask(block.mask_mod,1,1,1,kv_length,'cpu').reshape(-1)
                absolute=torch.arange(offset,length)
                expected=(absolute<=length-1)&(absolute!=0)
                if window is not None:expected &= absolute>length-1-window
                assert torch.equal(actual,expected),(length,window,cropped)
                # No true positions may be skipped by the compressed block map.
                dense=block.to_dense().reshape(-1).bool()
                for i in expected.nonzero().flatten().tolist():assert dense[i//128]
                cases+=1
assert masks.create_block_mask is original
print(f'PASS: {cases} causal/sliding-window/padding/cache-offset masks and context restoration')
