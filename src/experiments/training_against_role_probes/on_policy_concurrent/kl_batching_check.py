"""GPU check: batched reverse KL matches per-request replay, and how fast it is."""
import argparse
import json
from pathlib import Path
import random
import time

from peft import set_peft_model_state_dict
from safetensors.torch import load_file
import torch

from role_probe_training.on_policy.objective import kl_batches, reverse_kl_sum, reverse_kl_sums
from role_probe_training.training.data import read_jsonl
from role_probe_training.training.model import RoleModel


def length(request):
    return len(request['prompt_ids']) + len(request['completion_ids'])


def run(runtime, parameters, requests, max_tokens):
    """Accumulate KL gradients as the trainer does; return losses, gradient and cost."""
    for parameter in parameters:
        parameter.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for batch in kl_batches(requests, max_tokens):
        loss = reverse_kl_sums(runtime, batch)
        (loss/len(requests)).backward()
        losses.append(([r['key'] for r in batch], loss.item()))
        runtime.hidden = None
    torch.cuda.synchronize()
    gradient = torch.cat([p.grad.flatten().float().cpu() for p in parameters])
    return dict(losses=losses, gradient=gradient, seconds=time.perf_counter()-started,
                passes=len(losses), peak_gib=torch.cuda.max_memory_allocated()/2**30)


def compare(per_request, gradient, other):
    """Differences from the per-request run, in float64."""
    errors = []
    for keys, loss in other['losses']:
        expected = sum(per_request[k] for k in keys)
        errors.append(dict(expected=expected, absolute=abs(loss-expected)))
    worst = max(errors, key=lambda x: x['absolute']/max(abs(x['expected']), 1e-12))
    total = sum(per_request.values())
    a, b = gradient.double(), other['gradient'].double()
    return dict(kl_total_relative_error=abs(sum(loss for _, loss in other['losses'])-total)/total,
                max_pass_relative_error=worst['absolute']/max(abs(worst['expected']), 1e-12),
                worst_pass_expected_kl=worst['expected'],
                pass_absolute_error_sum=sum(x['absolute'] for x in errors),
                gradient_cosine=float((a @ b)/(a.norm()*b.norm())),
                gradient_relative_difference=float((a-b).norm()/a.norm()),
                gradient_norm_ratio=float(b.norm()/a.norm()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--count', type=int, default=96)
    parser.add_argument('--budgets', type=int, nargs='+', default=[32768, 65536])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = RoleModel(config['model'], config['probe'], 0)
    set_peft_model_state_dict(runtime.model, load_file(args.adapter))
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    pool = sorted(read_jsonl(args.requests), key=length)
    # Always include the longest contexts; fill the rest uniformly at random.
    chosen = pool[-4:] + random.Random(0).sample(pool[:-4], args.count-4)
    for index, request in enumerate(chosen):
        request['key'] = index
    warm = sorted(chosen, key=length)[:8]
    reverse_kl_sum(runtime, warm[0]).backward()
    reverse_kl_sums(runtime, warm[1:5]).backward()
    reverse_kl_sums(runtime, warm[5:8]).backward()
    single = run(runtime, parameters, chosen, None)
    per_request = {keys[0]: loss for keys, loss in single['losses']}
    result = dict(requests=len(chosen), tokens=sum(map(length, chosen)),
                  max_length=max(map(length, chosen)),
                  single=dict(seconds=single['seconds'], passes=single['passes'],
                              peak_gib=single['peak_gib'], kl_total=sum(per_request.values())))
    # bf16 GPU kernels are not deterministic: an identical rerun sets the noise floor.
    modes = [('single_repeat', None)] + [(f'budget_{budget}', budget) for budget in args.budgets]
    for name, budget in modes:
        other = run(runtime, parameters, chosen, budget)
        result[name] = dict(seconds=other['seconds'], passes=other['passes'], peak_gib=other['peak_gib'],
                            speedup=single['seconds']/other['seconds'],
                            **compare(per_request, single['gradient'], other))
        print(json.dumps({name: result[name]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
