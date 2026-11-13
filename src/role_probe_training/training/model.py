"""GPT-OSS LoRA model and correctly located differentiable role probe."""
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from role_probe_training.harmony import ROLE_SETS
from . import probe_lbfgs
from role_probe_training.probes.linear import RoleProbe
from .data import collate


class RoleModel:
    def __init__(self, model_path, probe_path, device, *, adapters=True):
        # Full histories, KL, and probe refits exercise distinct attention masks.
        # Never fall back to quadratic eager attention when the compile cache fills.
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.accumulated_recompile_limit = 512
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
            dtype=torch.bfloat16,device_map={'':device},quantization_config=Mxfp4Config(dequantize=True),
            attn_implementation='flex_attention')
        self.model=self.base
        if adapters:
            self.model=get_peft_model(self.base,LoraConfig(r=64,lora_alpha=128,lora_dropout=0.,bias='none',
                target_modules=['q_proj','k_proj','v_proj','o_proj'],
                target_parameters=['mlp.experts.gate_up_proj','mlp.experts.down_proj']))
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.model.enable_input_require_grads()
        else:
            self.model.requires_grad_(False)
        # Dropout is zero; train mode selects the memory-efficient routed-expert path.
        self.model.train()
        artifact=RoleProbe.load(probe_path)
        self.roles=tuple(artifact.metadata['roles'])  # the probe's role set labels all training data
        assert self.roles in ROLE_SETS.values()
        self.layer=artifact.metadata['layer']
        assert artifact.metadata['activation']=='post_attention_layernorm_output_pre_mlp'
        assert artifact.metadata['model_revision']=='6cee5e81ee83917806bbde320786a8fb61efebee'
        self.probe_metadata=artifact.metadata
        # A linear probe on raw activations, as stored. Training starts from the artifact's
        # weights, so the artifact must come from the project's fitter (probe_lbfgs) for the
        # first model update to see a converged probe.
        self.probe=torch.nn.Linear(artifact.weight.shape[1],len(self.roles),device=device,dtype=torch.float32)
        with torch.no_grad():
            self.probe.weight.copy_(torch.as_tensor(artifact.weight)); self.probe.bias.copy_(torch.as_tensor(artifact.bias))
        self.last_probe_refit=artifact.metadata.get('fit')
        self.probe.requires_grad_(False)
        self.hidden=None
        def capture(module,args,output): self.hidden=output
        self.handle=self.base.model.layers[self.layer].post_attention_layernorm.register_forward_hook(capture)

    def states(self, examples):
        ids,attention,labels,_=collate(examples,self.device,self.pad_id)
        if ids.shape[1]>self.base.config.max_position_embeddings:
            raise ValueError('Full trajectory exceeds model context; never silently truncate')
        self.hidden=None
        self.base.model(input_ids=ids,attention_mask=attention,use_cache=False)
        hidden=self.hidden; self.hidden=None
        return hidden,labels

    def role_loss(self, examples):
        hidden,labels=self.states(examples)
        valid=labels>=0
        logits=self.probe(hidden[valid].float())
        loss=F.cross_entropy(logits,labels[valid],reduction='sum')
        predicted=logits.argmax(-1)
        n=len(self.roles)
        counts=torch.bincount(labels[valid]*n+predicted,minlength=n*n).reshape(n,n)
        return loss,counts

    def save_probe(self,path,step):
        fit=self.last_probe_refit or {}
        RoleProbe(self.probe.weight.detach().cpu().numpy(),self.probe.bias.detach().cpu().numpy(),
            {**self.probe_metadata,'model_update_step':step,
             'refit':'warm-start full-batch L-BFGS on standard passages (probe_lbfgs)',
             'solver':'lbfgs','C':fit.get('C',probe_lbfgs.C),'fit':fit,
             'max_iter':fit.get('max_iterations'),'iterations':fit.get('iterations'),
             'converged':fit.get('converged'),'warnings':[]}).save(path)

    def close(self): self.handle.remove()
