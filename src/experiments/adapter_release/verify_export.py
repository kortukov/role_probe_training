"""Check all reconstructed weight tensors against the evaluated export."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
root=Path('experiment_reports/prompt_injection/agentdojo_same_batch_27000')
a=root/'data/adapter_reconstructed'
b=Path('experiment_reports/training_against_role_probes/data_iteration/same_batch_27000/data/post_training/model')
def index(path):
    return json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
ia,ib=index(a),index(b)
assert set(ia)==set(ib)
count=0
for shard in sorted(set(ia.values())):
    with safe_open(a/shard,framework='pt',device='cpu') as left:
        for key in left.keys():
            with safe_open(b/ib[key],framework='pt',device='cpu') as right:
                assert torch.equal(left.get_tensor(key),right.get_tensor(key)), key
            count+=1
with safe_open(a/'attention_adapter/adapter_model.safetensors',framework='pt',device='cpu') as left, safe_open(b/'attention_adapter/adapter_model.safetensors',framework='pt',device='cpu') as right:
    assert set(left.keys())==set(right.keys())
    for key in left.keys():
        assert torch.equal(left.get_tensor(key),right.get_tensor(key)),key
        count+=1
(root/'adapter_validation.json').write_text(json.dumps({'result':'PASS','compared_tensors':count,'comparison':'All reconstructed base/expert and separate attention tensors exactly equal the evaluated export'},indent=2)+'\n')
print(f'PASS: {count} tensors exactly equal the evaluated model')
