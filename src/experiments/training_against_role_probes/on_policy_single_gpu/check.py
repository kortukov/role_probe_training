"""CPU checks for single-rank collectives, summed gradients and probe refitting."""
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
from unittest.mock import patch
from role_probe_training.training.train_same_batch import sync_gradients
from role_probe_training.training.refit import refit_probe
from role_probe_training.training import probe_lbfgs

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
    torch.manual_seed(0)
    features=torch.randn(36,4,3); targets=torch.randint(0,5,(36,4))
    class Runtime:
        device='cpu'
        probe=torch.nn.Linear(3,5).requires_grad_(False)
        def states(self, examples):
            rows=[x['index'] for x in examples]
            return features[rows],targets[rows]
    examples=[dict(index=i,labels=targets[i].tolist(),split='train' if i<32 else 'test') for i in range(36)]
    runtime=Runtime(); start=runtime.probe.weight.clone()
    with patch('torch.cuda.synchronize'):
        result=refit_probe(runtime,examples)
    assert result['training_examples']==32 and result['training_tokens']==128 and result['tokens']==16
    # The refit is the project's L-BFGS fitter, warm-started from the current probe and run to
    # its gradient test; the same solve on the pooled tokens gives the same probe.
    assert result['solver']=='lbfgs' and result['converged'] and result['warm_start']
    assert result['gradient_max']<=result['gradient_tolerance'] and runtime.last_probe_refit['converged']
    assert not torch.equal(runtime.probe.weight,start) and not runtime.probe.weight.requires_grad
    weight,bias,_=probe_lbfgs.fit(features[:32].reshape(-1,3).half(),targets[:32].reshape(-1),classes=5)
    torch.testing.assert_close(runtime.probe.weight,weight,atol=1e-4,rtol=0)
    with patch('torch.cuda.synchronize'):
        again=refit_probe(runtime,examples)
    assert again['converged'] and again['iterations']<=1  # already at the minimum
    try:
        with patch('torch.cuda.synchronize'):
            runtime.probe.weight.zero_(); refit_probe(runtime,examples,max_iterations=1)
        raise AssertionError('An unconverged refit must stop training')
    except RuntimeError:
        pass
    dist.destroy_process_group()
print('PASS: world-size-one collectives, unchanged global gradient normalization, full refit coverage')
