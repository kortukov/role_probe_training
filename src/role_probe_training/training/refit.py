"""Warm-start probe refitting on all standard passage content, never agent data."""
import math
import random
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F


def summarize_counts(matrix):
    matrix=matrix.detach().cpu()
    support=matrix.sum(1)
    recalls=matrix.diag().double()/support.clamp(min=1)
    return dict(tokens=int(matrix.sum()),accuracy=float(matrix.diag().sum()/matrix.sum().clamp(min=1)),
                macro_recall=float(recalls[support>0].mean()),per_role_recall=recalls.tolist(),
                confusion_counts=matrix.tolist())


def refit_probe(runtime, examples, *, steps=2000, batch_size=32, seed=123):
    """A batch is 32 passage-role examples, including EVERY content token.

    Each shuffled pass visits every standard fitting example; there is no token
    subsampling. Split assignment remains the original passage-level assignment.
    Per-rank activation caches stay on their GPU and are freed before model updates.
    """
    rank=dist.get_rank(); world=dist.get_world_size()
    start=time.perf_counter(); features={}; labels={}
    owned=list(range(rank,len(examples),world))
    with torch.no_grad():
        for lo in range(0,len(owned),16):
            indices=owned[lo:lo+16]
            hidden,target=runtime.states([examples[i] for i in indices])
            for row,i in enumerate(indices):
                valid=target[row]>=0
                features[i]=hidden[row,valid].to(torch.float16)
                labels[i]=target[row,valid]
            if lo%160==0:
                print(f'Rank {rank}: extracted {min(lo+16,len(owned))}/{len(owned)} standard examples',flush=True)
    torch.cuda.synchronize(); extracted=time.perf_counter()-start
    fitting=[i for i,x in enumerate(examples) if x['split']=='train']
    heldout=[i for i,x in enumerate(examples) if x['split']!='train']
    if not fitting or not heldout: raise ValueError('Probe train/test split is missing')
    if steps and steps*batch_size<len(fitting):
        raise ValueError('Refitting budget must cover every standard fitting example')
    counts=[sum(y>=0 for y in x['labels']) for x in examples]
    runtime.probe.requires_grad_(True)
    optimizer=torch.optim.SGD(runtime.probe.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=random.Random(seed); order=[]; cursor=0
    last_loss=0.
    for step in range(steps):
        if cursor>=len(order):
            order=fitting.copy(); generator.shuffle(order); cursor=0
        batch=order[cursor:cursor+batch_size]; cursor+=len(batch)
        denominator=sum(counts[i] for i in batch)
        local=[i for i in batch if i%world==rank]
        optimizer.zero_grad(set_to_none=True)
        if local:
            x=torch.cat([features[i] for i in local]).float()
            y=torch.cat([labels[i] for i in local])
            loss=F.cross_entropy(runtime.probe(x),y,reduction='sum')/denominator
        else:
            loss=sum(p.sum()*0 for p in runtime.probe.parameters())
        loss.backward()
        for param in runtime.probe.parameters(): dist.all_reduce(param.grad)
        torch.nn.utils.clip_grad_norm_(runtime.probe.parameters(),1.)
        # Upstream refit schedule: 50-step linear warmup, then cosine decay.
        factor=(step+1)/50 if step<50 else .5*(1+math.cos(math.pi*(step-50)/max(1,steps-50)))
        optimizer.param_groups[0]['lr']=1e-3*factor
        optimizer.step()
        last_loss=loss.detach()
    runtime.probe.requires_grad_(False)
    confusion=torch.zeros((5,5),device=runtime.device,dtype=torch.long)
    with torch.no_grad():
        for i in heldout:
            if i not in features: continue
            prediction=runtime.probe(features[i].float()).argmax(-1)
            confusion+=torch.bincount(labels[i]*5+prediction,minlength=25).reshape(5,5)
    dist.all_reduce(confusion)
    torch.cuda.synchronize()
    return {**summarize_counts(confusion),'extraction_seconds':extracted,
            'seconds':time.perf_counter()-start,'steps':steps,'batch_passage_roles':batch_size,
            'training_examples':len(fitting),'training_tokens':sum(counts[i] for i in fitting)}
