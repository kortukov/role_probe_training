"""Warm-start probe refitting on all standard passage content, never agent data.

The optimisation is the fitter of training_against_probes (probe_fit.py): AdamW, its
cosine schedule and batching, a frozen input scale. Only the loss differs from upstream.
"""
import time
import torch
import torch.distributed as dist
from .probe_fit import fit, input_scale


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
    The probe continues from its current weights with a new optimizer and schedule.
    Its input scale stays as fixed before training; a probe whose artifact recorded none
    gets it here, once, from the fitting activations of the current model.
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
    if not bool(runtime.probe.scale_set):
        total=torch.zeros(2,dtype=torch.float64,device=runtime.device)
        for i in fitting:
            if i in features:
                total[0]+=features[i].double().square().sum(); total[1]+=features[i].shape[0]
        dist.all_reduce(total)
        runtime.probe.rescale(input_scale(total[0],total[1]))
    def batch(indices):
        local=[i for i in indices if i in features]
        if not local: return None
        return torch.cat([features[i] for i in local]).float(),torch.cat([labels[i] for i in local])
    rates=fit(runtime.probe,batch,fitting,counts,steps=steps,batch_size=batch_size,seed=seed,sync=dist.all_reduce)
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
            'optimizer':'AdamW','optimizer_steps':len(rates),'input_scale':float(runtime.probe.input_scale),
            'training_examples':len(fitting),'training_tokens':sum(counts[i] for i in fitting)}
