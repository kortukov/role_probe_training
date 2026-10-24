"""Serve a fixed FP32 attention adapter without rounding it into BF16 weights.

Expert deltas are already exactly merged in the exported base. This worker is
restricted to single-GPU inference, matching our benchmark job configuration.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from vllm.v1.worker.gpu_worker import Worker


class FixedAttentionAdapter(nn.Module):
    def __init__(self,base,pairs,scale):
        super().__init__();self.base=base;self.scale=scale;self.count=len(pairs)
        for i,(a,b) in enumerate(pairs):
            self.register_buffer(f'a{i}',a.float().contiguous())
            self.register_buffer(f'b{i}',b.float().contiguous())

    def forward(self,x):
        result,bias=self.base(x)
        assert bias is None, 'The base projection must include its bias'
        updates=[F.linear(F.linear(x.float(),getattr(self,f'a{i}')),getattr(self,f'b{i}'))*self.scale
                 for i in range(self.count)]
        return (result.float()+torch.cat(updates,dim=-1)).to(result.dtype),None


class RoleWorker(Worker):
    def load_model(self,*,load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        if self.parallel_config.tensor_parallel_size!=1 or self.parallel_config.pipeline_parallel_size!=1:
            raise ValueError('Fixed role-training adapters require single-GPU inference')
        torch.set_float32_matmul_precision('highest')
        path=Path(self.model_config.model)/'attention_adapter'
        config=json.loads((path/'adapter_config.json').read_text())
        assert not config['target_parameters']
        assert config['lora_dropout']==0 and not config['use_dora'] and not config['use_rslora']
        tensors=load_file(path/'adapter_model.safetensors',device=str(self.device))
        scale=config['lora_alpha']/config['r']; model=self.model_runner.get_model()
        for i,layer in enumerate(model.model.layers):
            def pair(name):
                prefix=f'base_model.model.model.layers.{i}.self_attn.{name}'
                return tensors[prefix+'.lora_A.weight'],tensors[prefix+'.lora_B.weight']
            layer.attn.qkv_proj=FixedAttentionAdapter(layer.attn.qkv_proj,[pair(k) for k in ('q_proj','k_proj','v_proj')],scale)
            layer.attn.o_proj=FixedAttentionAdapter(layer.attn.o_proj,[pair('o_proj')],scale)
        self.model_runner.model_memory_usage+=sum(v.numel()*v.element_size() for v in tensors.values())
        print('Installed fixed FP32 attention adapters on all decoder layers',flush=True)
