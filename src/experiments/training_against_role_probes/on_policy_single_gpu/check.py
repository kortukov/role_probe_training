"""CPU checks for single-rank collectives, summed gradients and probe refitting."""
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from unittest.mock import patch
from role_probe_training.training.train_same_batch import sync_gradients
from role_probe_training.training.refit import refit_probe

with tempfile.TemporaryDirectory() as folder:
    dist.init_process_group('gloo', init_method='file://'+str(Path(folder)/'rendezvous'), rank=0, world_size=1)
    for count in (14,16,64):
        x=torch.arange(count,dtype=torch.float64)/count
        p=torch.nn.Parameter(torch.tensor(0.3,dtype=torch.float64))
        for item in x:
            ((p*item-1).square()/count).backward()
        sync_gradients([p]); one=p.grad.clone()
        partial=[]
        for rank in (0,1):
            q=torch.nn.Parameter(p.detach().clone())
            ((q*x[rank::2]-1).square().sum()/count).backward()
            partial.append(q.grad)
        torch.testing.assert_close(one,sum(partial))
    class Runtime:
        device='cpu'
        probe=torch.nn.Linear(3,5)
        def states(self, examples):
            return torch.ones(len(examples),2,3),torch.tensor([[0,1]]*len(examples))
    examples=[dict(labels=[0,1],split='train' if i<32 else 'test') for i in range(36)]
    with patch('torch.cuda.synchronize'):
        result=refit_probe(Runtime(),examples,steps=2)
    assert result['training_examples']==32 and result['training_tokens']==64
    dist.destroy_process_group()
print('PASS: world-size-one collectives, unchanged global gradient normalization, full refit coverage')
