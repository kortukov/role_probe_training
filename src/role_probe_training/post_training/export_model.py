"""Prepare a checked BF16 expert checkpoint plus separate attention LoRA for vLLM."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from role_probe_training.training.model import RoleModel
from role_probe_training.training.data import trajectory_examples, collate


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-expert-export',action='store_true')
    a=p.parse_args()
    if a.output.exists() and not a.reuse_expert_export: raise FileExistsError(a.output)
    if a.reuse_expert_export:
        saved=json.loads((a.output/'export.json').read_text())
        assert saved['checkpoint']==str(a.checkpoint)
        assert all(x==0 for x in saved['activation_merge_relative_rms'])
    runtime=RoleModel('.cache/models/gpt-oss-20b',
        'experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz',0)
    adapter=a.checkpoint/'adapter/adapter_model.safetensors'
    set_peft_model_state_dict(runtime.model,load_file(adapter))
    runtime.model.gradient_checkpointing_disable()
    examples=trajectory_examples(runtime.tokenizer,
        'experiment_reports/training_against_role_probes/runs/dev/1501280/dev/trajectories.jsonl')[:2]
    examples=[{**x,'ids':x['ids'][:1024],'labels':x['labels'][:1024]} for x in examples]
    def snapshot():
        result=[]
        with torch.no_grad():
            for example in examples:
                ids,attention,_,_=collate([example],0,runtime.pad_id)
                last=runtime.base.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
                result.append((runtime.hidden.float().cpu(),runtime.base.lm_head(last[:,-1]).float().cpu()))
                runtime.hidden=None
        return result
    before=snapshot()
    # ParamWrapper already forms W + delta in BF16 inside every training forward.
    for layer in runtime.base.model.layers:
        layer.mlp.experts=layer.mlp.experts.unload_and_optionally_merge_module(
            merge=True,safe_merge=True,adapter_names=['default'])
    expert=snapshot()
    exact=[float((x[0]-y[0]).square().mean().sqrt()/x[0].square().mean().sqrt()) for x,y in zip(before,expert)]
    if not all(v<1e-5 for v in exact):raise ValueError(f'Expert merge must be exact: {exact}')
    kls=[float(torch.nn.functional.kl_div(y[1].log_softmax(-1), x[1].softmax(-1), reduction='batchmean'))
         for x,y in zip(before,expert)]
    print(json.dumps({'expert_merge_relative_rms':exact}),flush=True)
    runtime.close(); model=runtime.model.unload()
    if hasattr(model.config,'quantization_config'):del model.config.quantization_config
    if hasattr(model,'hf_quantizer'):del model.hf_quantizer
    model.is_quantized=False
    if not a.reuse_expert_export:
        a.output.mkdir(parents=True)
        model.save_pretrained(a.output,safe_serialization=True,max_shard_size='4GB')
        runtime.tokenizer.save_pretrained(a.output)
    attention_path=a.output/'attention_adapter';attention_path.mkdir(exist_ok=a.reuse_expert_export)
    tensors=load_file(adapter)
    attention={k:v for k,v in tensors.items() if '.self_attn.' in k}
    assert len(attention)==24*8
    save_file(attention,str(attention_path/'adapter_model.safetensors'))
    config=json.loads((a.checkpoint/'adapter/adapter_config.json').read_text())
    config['target_parameters']=None;config['base_model_name_or_path']=str(a.output.resolve())
    (attention_path/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
    metadata={'checkpoint':str(a.checkpoint),'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),
        'base_revision':'6cee5e81ee83917806bbde320786a8fb61efebee','dtype':'bfloat16',
        'expert_merge_relative_rms':exact,'expert_merge_next_token_kl':kls,
        'attention_adapter':'attention_adapter','attention_dtype':'float32',
        'note':'Expert deltas merged exactly; separate FP32 attention adapters used for both probing and generation.'}
    (a.output/'export.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
