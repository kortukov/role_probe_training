"""CPU checks for KL direction, masking, checkpointed gradients and padding."""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from role_probe_training.training.data import collate
from role_probe_training.training.objectives import kl_sum


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(11,7).requires_grad_(False)
        self.adapter=torch.nn.Parameter(torch.randn(7)*.05)
        self.active=True
    def forward(self,input_ids,**kwargs):
        hidden=self.embedding(input_ids)
        if self.active: hidden=hidden+self.adapter
        return SimpleNamespace(last_hidden_state=hidden)
    @contextmanager
    def disable_adapter(self):
        self.active=False
        try: yield
        finally: self.active=True


def main():
    torch.manual_seed(1)
    backbone=ToyBackbone()
    head=torch.nn.Linear(7,11,bias=False).requires_grad_(False)
    runtime=SimpleNamespace(model=backbone,base=SimpleNamespace(model=backbone,lm_head=head),device='cpu',pad_id=0)
    examples=[dict(ids=[1,2,3,4],completion_mask=[False,False,True,True]),
              dict(ids=[4,5,6],completion_mask=[False,True,True])]
    ids,attention,_,mask=collate(examples,'cpu',0)
    assert mask.sum()==4 and attention.sum()==7 and not mask[1,-1]
    actual=kl_sum(runtime,examples,chunk_size=2)
    actual.backward(); gradient=backbone.adapter.grad.clone(); backbone.adapter.grad=None
    with backbone.disable_adapter(),torch.no_grad():
        p=F.softmax(head(backbone(ids).last_hidden_state).float(),-1)
    logq=F.log_softmax(head(backbone(ids).last_hidden_state).float(),-1)
    expected=(F.kl_div(logq,p,reduction='none').sum(-1)*mask).sum()
    expected.backward()
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(gradient,backbone.adapter.grad,atol=1e-6,rtol=1e-5)
    with torch.no_grad(): backbone.adapter.zero_()
    torch.testing.assert_close(kl_sum(runtime,examples),torch.tensor(0.),atol=1e-6,rtol=0)
    role=[dict(ids=[1,2],labels=[-1,3]),dict(ids=[1],labels=[4])]
    _,attention,labels,_=collate(role,'cpu',0)
    assert labels.tolist()==[[-1,3],[4,-1]] and attention.tolist()==[[1,1],[1,0]]
    print('KL direction, completion mask, checkpointed gradients, zero-adapter KL and role padding passed.')

if __name__=='__main__': main()
