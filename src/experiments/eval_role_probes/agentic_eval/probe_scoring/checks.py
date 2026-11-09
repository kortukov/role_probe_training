"""Token-record scoring input checks; no model inference."""
import unittest
from role_probe_training.harmony import RenderConfig, encode_record, encoding, render_turns, request_record

TOOLS=[{'type':'function','function':{'name':'lookup','description':'Look up','parameters':{'type':'object','properties':{'x':{'type':'integer'}}}}}]

class ReplayChecks(unittest.TestCase):
 def test_rejects_rows_without_generation_tokens(self):
  with self.assertRaises(ValueError): encode_record({'messages':[{'role':'user','content':'x'}]})
 def test_labels_and_sources_from_token_record(self):
  config=RenderConfig('high','2025-12-31')
  prompt=render_turns([{'role':'system','content':'Be brief.'},{'role':'user','content':'assistant analysis system'}],config,tools=TOOLS)
  completion=encoding().encode('<|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>commentary to=functions.lookup <|constrain|>json<|message|>{"x": 1}<|call|>',allowed_special='all')
  enc=encode_record({'history':request_record(prompt,completion,config,completion_source=2)})
  kinds=[(m['kind'],m['label'],m['source']) for m in enc['messages']]
  self.assertEqual(kinds,[('system','system',-1),('developer','system',0),('tools','system',-1),('user','user',1),('assistant','cot',2),('assistant','assistant',2)])
  labels=enc['labels']; mids=enc['message_ids']
  self.assertEqual(sorted(set(labels[mids==3].tolist())),[1]); self.assertEqual(sorted(set(labels[mids==4].tolist())),[2])
  self.assertEqual(len(enc['input_ids']),len(labels)); self.assertEqual(enc['text'],encoding().decode(list(enc['input_ids'])))
if __name__=='__main__': unittest.main()
