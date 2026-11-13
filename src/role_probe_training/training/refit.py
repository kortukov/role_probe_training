"""Warm-start probe refitting on all standard passage content, never agent data.

The fitter is the project's one probe fitter (probe_lbfgs.py): full-batch L-BFGS on the
role-probe paper's objective, run to its gradient test. A refit starts from the current
probe; a refit that does not converge stops training.
"""
import time
import torch
import torch.distributed as dist
from . import probe_lbfgs


def summarize_counts(matrix):
    matrix=matrix.detach().cpu()
    support=matrix.sum(1)
    recalls=matrix.diag().double()/support.clamp(min=1)
    return dict(tokens=int(matrix.sum()),accuracy=float(matrix.diag().sum()/matrix.sum().clamp(min=1)),
                macro_recall=float(recalls[support>0].mean()),per_role_recall=recalls.tolist(),
                confusion_counts=matrix.tolist())


def refit_probe(runtime, examples, *, C=probe_lbfgs.C, gradient_tolerance=probe_lbfgs.GRADIENT_TOLERANCE,
                max_iterations=probe_lbfgs.MAX_ITERATIONS):
    """Refit on EVERY content token of every standard fitting example; no subsampling.

    Split assignment remains the original passage-level assignment. Each process extracts
    and keeps the activations of its own examples on its GPU; loss and gradient are summed
    over processes, so all follow the same L-BFGS iterates. Caches are freed before model
    updates. The result is stored as `runtime.last_probe_refit` for the probe checkpoint.
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
    tokens=sum(sum(y>=0 for y in examples[i]['labels']) for i in fitting)
    local=[i for i in fitting if i in features]
    hidden_size=runtime.probe.in_features
    x=torch.cat([features[i] for i in local]) if local else torch.empty((0,hidden_size),dtype=torch.float16,device=runtime.device)
    y=torch.cat([labels[i] for i in local]) if local else torch.empty(0,dtype=torch.long,device=runtime.device)
    weight,bias,fit=probe_lbfgs.fit(x,y,classes=runtime.probe.out_features,weight=runtime.probe.weight.detach(),
        bias=runtime.probe.bias.detach(),C=C,tokens=tokens,sync=dist.all_reduce,gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations,log=probe_lbfgs.print_log if rank==0 else None)
    del x,y
    if not fit['converged']:
        raise RuntimeError(f'Probe refit did not converge: {fit}')
    with torch.no_grad():
        runtime.probe.weight.copy_(weight); runtime.probe.bias.copy_(bias)
    classes=runtime.probe.out_features
    confusion=torch.zeros((classes,classes),device=runtime.device,dtype=torch.long)
    with torch.no_grad():
        for i in heldout:
            if i not in features: continue
            prediction=runtime.probe(features[i].float()).argmax(-1)
            confusion+=torch.bincount(labels[i]*classes+prediction,minlength=classes*classes).reshape(classes,classes)
    dist.all_reduce(confusion)
    torch.cuda.synchronize()
    runtime.last_probe_refit=fit
    return {**summarize_counts(confusion),**fit,'extraction_seconds':extracted,
            'seconds':time.perf_counter()-start,'training_examples':len(fitting)}
