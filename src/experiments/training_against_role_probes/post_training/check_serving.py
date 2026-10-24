"""Compare original PEFT checkpoint and deployed model on identical token prefixes."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument('stage',choices=['reference','compare'])
p.add_argument('--root',type=Path,required=True)
p.add_argument('--base-url');p.add_argument('--model')
a=p.parse_args();target=a.root/'data/post_training/serving_check.json'
if a.stage=='reference':
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from role_probe_training.training.model import RoleModel
    from role_probe_training.training.data import trajectory_examples,collate
    export=json.loads((a.root/'data/post_training/model/export.json').read_text())
    runtime=RoleModel('.cache/models/gpt-oss-20b','experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz',0)
    set_peft_model_state_dict(runtime.model,load_file(Path(export['checkpoint'])/'adapter/adapter_model.safetensors'))
    runtime.model.gradient_checkpointing_disable()
    examples=trajectory_examples(runtime.tokenizer,'experiment_reports/training_against_role_probes/runs/dev/1501280/dev/trajectories.jsonl')[:2]
    rows=[]
    with torch.no_grad():
        for x in examples:
            ids=x['ids'][:1024]
            last=runtime.base.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False).last_hidden_state[:,-1]
            logprobs=runtime.base.lm_head(last).float().log_softmax(-1)[0]
            rows.append({'ids':ids,'logprobs':logprobs.cpu().tolist(),'top_token':int(logprobs.argmax())})
    target.write_text(json.dumps(rows))
else:
    from openai import OpenAI
    client=OpenAI(base_url=a.base_url,api_key='local')
    rows=json.loads(target.read_text());results=[]
    for row in rows:
        response=client.completions.create(model=a.model,prompt=row['ids'],max_tokens=1,temperature=0,logprobs=20,
            extra_body={'return_tokens_as_token_ids':True,'return_token_ids':True})
        choice=response.choices[0];top=choice.logprobs.top_logprobs[0]
        diffs=[]
        for name,value in top.items():
            token=int(name.removeprefix('token_id:'))
            diffs.append(abs(value-row['logprobs'][token]))
        emitted=int(choice.logprobs.tokens[0].removeprefix('token_id:'))
        results.append({'same_greedy_token':emitted==row['top_token'],'hf_top_token':row['top_token'],
            'served_top_token':emitted,'max_top20_logprob_difference':max(diffs),
            'mean_top20_logprob_difference':sum(diffs)/len(diffs)})
    (target.parent/'serving_check_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)
    assert all(r['same_greedy_token'] and r['mean_top20_logprob_difference']<0.2 for r in results),results
