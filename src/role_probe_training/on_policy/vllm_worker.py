"""Experimental synchronous refresh for the single-GPU BF16 Triton worker.

Node-local safetensors staging deliberately avoids cross-version CUDA IPC in the
first validation. Only expert parameters and FP32 attention LoRA change.
"""
from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from role_probe_training.post_training.vllm_worker import RoleWorker


class RefreshWorker(RoleWorker):
    @torch.no_grad()
    def refresh_role_policy(self, directory):
        root = Path(directory)
        metadata = json.loads((root/'manifest.json').read_text())
        model = self.model_runner.get_model()
        verified = 0
        for index, layer in enumerate(model.model.layers):
            weights = load_file(root/f'experts{index}.safetensors', device=str(self.device))
            model.load_weights(iter(weights.items()))
            # This explicit check rejects any unexpected kernel weight layout.
            experts = layer.mlp.experts.routed_experts
            for name, target in [('gate_up_proj', experts.w13_weight),
                                 ('down_proj', experts.w2_weight)]:
                expected = weights[f'model.layers.{index}.mlp.experts.{name}'].permute(0, 2, 1)
                if not torch.equal(target, expected):
                    raise ValueError(f'Expert refresh mismatch at layer {index}: {name}')
                verified += 1
            del weights
        tensors = load_file(root/'attention.safetensors', device=str(self.device))
        for index, layer in enumerate(model.model.layers):
            for projection, names in [(layer.attn.qkv_proj, ('q_proj','k_proj','v_proj')),
                                       (layer.attn.o_proj, ('o_proj',))]:
                for pair, name in enumerate(names):
                    prefix = f'base_model.model.model.layers.{index}.self_attn.{name}'
                    for side, suffix in [('a','A'),('b','B')]:
                        target = getattr(projection, f'{side}{pair}')
                        source = tensors[f'{prefix}.lora_{suffix}.weight']
                        target.copy_(source)
                        if not torch.equal(target, source):
                            raise ValueError('Attention refresh mismatch')
                        verified += 1
        torch.cuda.synchronize()
        return dict(policy_step=metadata['policy_step'], verified_tensors=verified)
