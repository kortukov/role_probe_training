"""Download an adapter and its public base; reconstruct the evaluated vLLM export."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = 'kortukov/gpt-oss-20b-role-perception-appworld-27000'
REVISION = '6cee5e81ee83917806bbde320786a8fb61efebee'


def prepare(adapter, output, base='openai/gpt-oss-20b'):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    adapter = Path(adapter)
    config = json.loads((adapter/'adapter_config.json').read_text())
    digest = hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
    if output.exists():
        saved = json.loads((output/'export.json').read_text())
        if saved['adapter_sha256'] != digest or saved['base_revision'] != REVISION:
            raise ValueError('Existing export belongs to another checkpoint')
        return
    output.mkdir(parents=True)
    torch.set_float32_matmul_precision('highest')
    tokenizer = AutoTokenizer.from_pretrained(base, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(base, revision=REVISION,
        dtype=torch.bfloat16, device_map={'': 0},
        quantization_config=Mxfp4Config(dequantize=True), attn_implementation='flex_attention')
    wrapped = PeftModel.from_pretrained(model, adapter, is_trainable=False, autocast_adapter_dtype=True)
    # PEFT's ParamWrapper forms W+dW in BF16 on each forward. Its merge
    # reproduces that operation; attention adapters must stay separate in FP32.
    for layer in model.model.layers:
        layer.mlp.experts = layer.mlp.experts.unload_and_optionally_merge_module(
            merge=True, safe_merge=True, adapter_names=['default'])
    model = wrapped.unload()
    for name in ('quantization_config',):
        if hasattr(model.config, name):
            delattr(model.config, name)
    if hasattr(model, 'hf_quantizer'):
        delattr(model, 'hf_quantizer')
    model.is_quantized = False
    model.save_pretrained(output, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(output)
    attention = {k:v for k,v in load_file(adapter/'adapter_model.safetensors').items() if '.self_attn.' in k}
    assert len(attention) == 192
    path = output/'attention_adapter'
    path.mkdir()
    save_file(attention, str(path/'adapter_model.safetensors'))
    config.update(target_parameters=None, base_model_name_or_path=str(output.resolve()), task_type=None)
    (path/'adapter_config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'export.json').write_text(json.dumps({'adapter_sha256':digest, 'base_revision':REVISION,
        'dtype':'bfloat16', 'attention_dtype':'float32'}, indent=2)+'\n')


def main():
    from huggingface_hub import snapshot_download
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--adapter', default=REPO, help='Hub repository or local adapter directory')
    p.add_argument('--base', default='openai/gpt-oss-20b')
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    adapter = a.adapter if Path(a.adapter).is_dir() else snapshot_download(a.adapter)
    prepare(adapter, a.output, a.base)

if __name__ == '__main__':
    main()
