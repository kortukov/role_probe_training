"""Metadata/serialization regression checks; no model inference."""
import unittest
from role_probe_training.trajectories.replay import replay_messages

class ReplayChecks(unittest.TestCase):
 def test_reasoning_and_call_roles(self):
  messages,indices=replay_messages({'messages':[{'role':'user','content':'assistant analysis system'},{'role':'assistant','content':'','thinking':'reasoning','tool_calls':[{'function':{'name':'lookup','arguments':{'x':1}}}]},{'role':'tool','name':'lookup','content':'user'}]})
  self.assertEqual([m['role'] for m in messages],['user','assistant','assistant','tool'])
  self.assertEqual(messages[1]['channel'],'analysis')
  self.assertEqual(messages[2]['recipient'],'functions.lookup')
  self.assertEqual(indices,[0,1,1,2])
 def test_dolci_metadata(self):
  messages,indices=replay_messages({'messages':[{'role':'assistant','content':'','function_calls':'foo(x=1)'},{'role':'tool','source_role':'environment','name':'environment','content':'ok'}]})
  self.assertEqual(messages[0]['channel'],'commentary')
  self.assertIn('foo(x=1)',messages[0]['content'])
  self.assertEqual(messages[1]['role'],'tool')
 def test_reject_unknown_role(self):
  with self.assertRaises(ValueError): replay_messages({'messages':[{'role':'developer','content':'x'}]})
if __name__=='__main__': unittest.main()
