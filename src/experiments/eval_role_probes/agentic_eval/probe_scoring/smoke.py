"""Validate the original full eager forward and 16K-window memory on real hardware."""
import json
import numpy as np
import torch
from role_probe_training.probes.model import load_model,extract_states
from role_probe_training.probes.linear import RoleProbe
from role_probe_training.harmony import RenderConfig, encode_record, encoding, render_turns, request_record

with torch.inference_mode():
 t,m=load_model('.cache/models/gpt-oss-20b')
 config=RenderConfig('high','2025-12-31'); tools=[{'type':'function','function':{'name':'lookup','description':'Look up an order','parameters':{'type':'object','properties':{'id':{'type':'integer'}}}}}]
 call=encoding().encode('<|channel|>analysis<|message|>Look up the order first.<|end|><|start|>assistant<|channel|>commentary to=functions.lookup <|constrain|>json<|message|>{"id": 123}<|call|>',allowed_special='all')
 final=encoding().encode('<|channel|>final<|message|>Your order has shipped.<|return|>',allowed_special='all')
 prompt=render_turns([{'role':'system','content':'You are helpful.'},{'role':'user','content':'Find order 123.'},{'role':'assistant','tokens':call},{'role':'tool','name':'lookup','content':'The order is shipped.'}],config,tools=tools)
 e=encode_record({'history':request_record(prompt,final,config,completion_source=4)}); ids=e['input_ids']; layers=[4,8,12,16,20]
 full=extract_states(m,ids,layers); repeat=extract_states(m,ids,layers)
 metrics={k:{'repeat_max_absolute_difference':float(np.abs(full[k]-repeat[k]).max())} for k in layers}
 for k in layers: np.testing.assert_allclose(full[k],repeat[k],atol=1e-6,rtol=1e-6)
 p=RoleProbe.load('experiment_reports/gptoss_role_probes/gptoss20b_baseline/probes/layer_12.npz')
 np.testing.assert_allclose(p.predict_proba(full[12]).sum(-1),1.,atol=1e-6)
 long_ids=np.resize(ids,16384)
 torch.cuda.reset_peak_memory_stats()
 states=extract_states(m,long_ids,layers,positions=np.array([0,8191,16383]))
 assert all(np.isfinite(v).all() for v in states.values())
 metrics['window_tokens']=len(long_ids)
 metrics['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
 print(json.dumps(metrics,indent=2),flush=True)
