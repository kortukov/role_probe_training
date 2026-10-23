"""Generate a fixed base GPT-OSS KL anchor, retaining exact Harmony token IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    model_path='.cache/models/gpt-oss-20b'
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    prompts=json.loads(a.input.read_text())
    ids=[tokenizer.apply_chat_template([{'role':'user','content':q}],tokenize=True,
              add_generation_prompt=True,reasoning_effort='high',return_dict=False) for q in prompts]
    assert all(isinstance(x,list) and all(isinstance(t,int) for t in x) for x in ids)
    if max(map(len,ids))+1536>32768:
        raise ValueError('KL prompt exceeds generation context; do not silently truncate')
    started=time.perf_counter()
    model=LLM(model=model_path,max_model_len=32768,max_num_seqs=16,gpu_memory_utilization=.80,
              attention_backend='TRITON_ATTN',moe_backend='triton',
              kernel_config={'enable_jit_warmup':False},
              compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16],'max_cudagraph_capture_size':16})
    outputs=model.generate([{'prompt_token_ids':x} for x in ids],
              SamplingParams(temperature=0,max_tokens=1536,seed=42),use_tqdm=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for i,(prompt,prefix,out) in enumerate(zip(prompts,ids,outputs)):
            completion=list(out.outputs[0].token_ids)
            f.write(json.dumps(dict(id=i,prompt=prompt,prompt_ids=prefix,completion_ids=completion,
                completion=tokenizer.decode(completion,skip_special_tokens=False),
                finish_reason=out.outputs[0].finish_reason))+'\n')
    a.output.with_suffix('.manifest.json').write_text(json.dumps(dict(samples=len(outputs),
        seconds=time.perf_counter()-started,model=model_path,max_new_tokens=1536,
        temperature=0,seed=42,prompts_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__': main()
