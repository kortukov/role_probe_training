"""Score complete fixed histories with existing probes; never fit or update weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from role_probe_training.probes.harmony import ROLES
from role_probe_training.probes.linear import RoleProbe
from role_probe_training.trajectories.replay import encode_record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='.cache/models/gpt-oss-20b')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(a.input.read_bytes()).hexdigest()
    manifest = {'input_sha256': digest, 'input': str(a.input), 'roles': ROLES,
        'model_revision': '6cee5e81ee83917806bbde320786a8fb61efebee',
        'activation': 'post_attention_layernorm_output_pre_mlp',
        'precision': 'MXFP4 dequantized to BF16', 'attention': 'flex_attention',
        'context': 'complete retrospective Harmony; no truncation or window reset',
        'mask': 'original content only; exclude headers, boundaries, synthetic tool schemas',
        'job_id': os.environ.get('SLURM_JOB_ID'), 'limit': a.limit}
    mp = a.output/'manifest.json'
    if mp.exists():
        old = json.loads(mp.read_text())
        assert old['input_sha256'] == digest and old['limit'] == a.limit
    probes = {}
    for layer in [4, 8, 12, 16, 20]:
        folder = 'gptoss20b_baseline' if layer in [12, 16] else 'gptoss20b_layers'
        path = Path(f'experiment_reports/role_probes/{folder}/probes/layer_{layer}.npz')
        probe = RoleProbe.load(path)
        assert probe.metadata['roles'] == list(ROLES)
        assert probe.metadata['model_revision'] == manifest['model_revision']
        assert probe.metadata['activation'] == manifest['activation']
        assert probe.metadata['layer'] == layer
        probes[layer] = probe
    manifest['probes'] = {str(k): v.metadata for k, v in probes.items()}
    mp.write_text(json.dumps(manifest, indent=2)+'\n')
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        dtype=torch.bfloat16, device_map='cuda', quantization_config=Mxfp4Config(dequantize=True),
        attn_implementation='flex_attention').eval()
    model.requires_grad_(False)
    weights = {k: torch.tensor(v.weight, device='cuda', dtype=torch.float32) for k, v in probes.items()}
    biases = {k: torch.tensor(v.bias, device='cuda', dtype=torch.float32) for k, v in probes.items()}
    captured = {}
    def hook(layer):
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = torch.softmax(hidden[0].float() @ weights[layer].T + biases[layer], -1).cpu().numpy()
        return capture
    handles = [model.model.layers[k].post_attention_layernorm.register_forward_hook(hook(k)) for k in probes]
    records_path = a.output/'per_trajectory.jsonl'
    results = [json.loads(line) for line in records_path.open()] if records_path.exists() else []
    done = {r['trajectory_id'] for r in results}
    with records_path.open('a', buffering=1) as out, torch.inference_mode():
        for i, line in enumerate(a.input.open()):
            if a.limit is not None and i >= a.limit:
                break
            row = json.loads(line)
            identifier = row.get('trajectory_id', row['sample_id'])
            if identifier in done:
                continue
            start = time.monotonic()
            enc = encode_record(tokenizer, row)
            labels = enc['labels'].copy()
            for j, mid in enumerate(enc['message_ids']):
                if mid < 0 or enc['source_message_indices'][int(mid)] < 0:
                    labels[j] = -1
            n = len(labels)
            if n > model.config.max_position_embeddings:
                raise ValueError(f'{identifier}: {n} tokens exceeds full-context limit')
            ids = torch.tensor(enc['input_ids'], device='cuda')[None, :]
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            valid = labels >= 0
            y = labels[valid]
            layer_results = {}
            for k, probabilities in captured.items():
                prob = probabilities[valid]
                pred = prob.argmax(-1)
                cm = np.zeros((5, 5), dtype=np.int64)
                np.add.at(cm, (y, pred), 1)
                layer_results[str(k)] = {'confusion_matrix': cm.tolist(),
                    'true_probability_sums': [float(prob[y == c, c].sum()) for c in range(5)]}
            result = {'trajectory_id': identifier, 'sample_id': row['sample_id'],
                'required_calls': row.get('required_calls'), 'tokens': n,
                'layers': layer_results, 'seconds': time.monotonic()-start}
            out.write(json.dumps(result)+'\n')
            results.append(result)
            done.add(identifier)
            print(f'{i}: {identifier} {n} tokens {result["seconds"]:.1f}s', flush=True)
    for handle in handles:
        handle.remove()
    summary = {'trajectories': len(results), 'input_sha256': digest,
        'tokens': {'total': sum(r['tokens'] for r in results),
            'median': float(np.median([r['tokens'] for r in results])),
            'p95': float(np.percentile([r['tokens'] for r in results], 95)),
            'max': max(r['tokens'] for r in results)}, 'layers': {}}
    for k in map(str, probes):
        cm = sum(np.array(r['layers'][k]['confusion_matrix']) for r in results)
        counts = cm.sum(1)
        recall = np.divide(cm.diagonal(), counts, out=np.zeros(5), where=counts > 0)
        prob = sum(np.array(r['layers'][k]['true_probability_sums']) for r in results)
        summary['layers'][k] = {'confusion_matrix': cm.tolist(), 'scored_tokens': int(cm.sum()),
            'token_accuracy': float(cm.trace()/cm.sum()), 'macro_recall': float(recall[counts > 0].mean()),
            'majority_role_accuracy': float(counts.max()/counts.sum()),
            'trajectory_accuracy': float(np.mean([np.trace(r['layers'][k]['confusion_matrix'])/
                np.sum(r['layers'][k]['confusion_matrix']) for r in results])),
            'roles': {role: {'tokens': int(counts[c]), 'recall': float(recall[c]) if counts[c] else None,
                'mean_true_probability': float(prob[c]/counts[c]) if counts[c] else None}
                for c, role in enumerate(ROLES)}}
    (a.output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
