"""Use standard Transformers forwards; capture post-attention-normalized pre-MLP states."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=Mxfp4Config(dequantize=True), attn_implementation="eager",
    ).eval()
    model.requires_grad_(False)
    return tokenizer, model


@torch.inference_mode()
def extract_states(model, input_ids, layers, positions=None):
    """Layer 12 means output of layers[12].post_attention_layernorm (MLP input).

    This matches upstream all_pre_mlp_hidden_states; output_hidden_states
    exposes a different activation site and must not be used for these probes. Only requested tokens are copied to host memory.
    """
    if any(layer < 0 or layer >= len(model.model.layers) for layer in layers):
        raise ValueError("Invalid zero-based decoder block index")
    states, handles = {}, []
    if positions is None:
        positions = np.arange(len(input_ids))
    pos = torch.as_tensor(positions, device=model.device)
    def capture(layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[layer] = hidden[0, pos].to(dtype=torch.float32).cpu().numpy()
        return hook
    try:
        for layer in layers:
            handles.append(model.model.layers[layer].post_attention_layernorm.register_forward_hook(capture(layer)))
        ids = torch.as_tensor(input_ids, device=model.device).unsqueeze(0)
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return states
