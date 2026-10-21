"""Check causal chunked replay against a full forward on the actual model."""
import json
import numpy as np
import torch
from role_probe_training.probes.model import load_model,extract_states
from role_probe_training.probes.linear import RoleProbe
from role_probe_training.trajectories.replay import encode_record

with torch.inference_mode():
 t,m=load_model('.cache/models/gpt-oss-20b')
 r={'messages':[{'role':'system','content':'You are helpful.'},{'role':'user','content':'Find order 123.'},{'role':'assistant','thinking':'Look up the order first.','content':'','tool_calls':[{'function':{'name':'lookup','arguments':{'id':123}}}]},{'role':'tool','name':'lookup','content':'The order is shipped.'},{'role':'assistant','content':'Your order has shipped.'}]}
 e=encode_record(t,r); ids=e['input_ids']; layers=[4,8,12,16,20]
 full=extract_states(m,ids,layers)
 cached={k:[] for k in layers}
 def capture(k):
  def fn(module,args,output): cached[k].append(output[0].float().cpu().numpy())
  return fn
 handles=[m.model.layers[k].post_attention_layernorm.register_forward_hook(capture(k)) for k in layers]
 cache=None
 for lo in range(0,len(ids),17):
  hi=min(len(ids),lo+17)
  out=m.model(input_ids=torch.as_tensor(ids[lo:hi],device=m.device)[None,:],attention_mask=torch.ones((1,hi),device=m.device,dtype=torch.long),past_key_values=cache,use_cache=True)
  cache=out.past_key_values
 for h in handles: h.remove()
 metrics={}
 for k in layers:
  a=np.concatenate(cached[k]); b=full[k]
  metrics[k]={'max_absolute_state_difference':float(np.abs(a-b).max()),'relative_rms':float(np.sqrt(np.mean((a-b)**2))/np.sqrt(np.mean(b**2)))}
  assert metrics[k]['relative_rms']<0.02,metrics[k]
 p=RoleProbe.load('experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz')
 a=p.predict_proba(np.concatenate(cached[12])); b=p.predict_proba(full[12])
 metrics['probe_max_probability_difference']=float(np.abs(a-b).max())
 metrics['probe_argmax_agreement']=float((a.argmax(-1)==b.argmax(-1)).mean())
 assert metrics['probe_max_probability_difference']<0.05
 print(json.dumps(metrics,indent=2))
