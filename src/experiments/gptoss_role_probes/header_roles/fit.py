"""Header-role probes on the C4/Dolma passages, fitted to convergence with both fitters.

    fit.py extract   render every passage in every header role, capture the pre-MLP states of
                     the configured layers and cache them (ignored artifacts/ folder)
    fit.py fit       per layer: fit with cuML and with the AdamW fitter until each passes its
                     convergence test; save probes, loss histories, train and validation metrics
    fit.py compare   per layer: how alike the cuML header probe and each AdamW header probe are
                     on the validation tokens (runs on a CPU)

The stages are separate processes so that the model is not on the GPU during fitting and a
fit can be repeated without another extraction. Finished fits are skipped on a rerun; an
AdamW fit that has not converged continues with the longer schedules added to the config.

Convergence tests (settings in config.json):

* cuML solves a fixed objective, mean cross-entropy plus an L2 penalty, and stops by its own
  criteria with the configured `tol`: the gradient test, the loss-change test, the iteration
  limit (`max_iter`) or a failed line search (100 trial steps). A fit counts as converged when
  it stopped on the gradient or the loss-change test. The gradient of the objective at the
  solution is recomputed in double precision and reported, but decides nothing.
* The AdamW fitter runs a fixed number of steps on a cosine schedule that ends at rate zero,
  so its loss always flattens at the end. A fit counts as converged when multiplying the
  number of steps by the next factor in `steps` changes the final training loss by less
  than `relative_tolerance`.

`fit --backend adamw --learning-rate R` runs the same AdamW fitter with peak learning rate R
instead of upstream's 1e-3 and keeps its results beside the upstream-rate fit (file and
folder names end in `_lrR`). `fit --backend cuml --tol T` does the same for a cuML tolerance
other than the configured one (names end in `_cuml_tolT`).
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from role_probe_training.harmony import LIBRARY, ROLE_SETS, RenderConfig, document_header, render_document
from role_probe_training.probes.evaluation import metrics
from role_probe_training.probes.linear import RoleProbe, fit_probe

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ITERATION = re.compile(r'(\d+): f\(x\)=(\S+) conv\.crit=(\S+) \(gnorm=([^,\s)]+)')
TERMINATIONS = (('Converged after', 'gradient test'), ('Insufficient change in objective value', 'loss-change test'),
                ('max iterations reached', 'iteration limit'), ('line search failed', 'line search failed'),
                ('failed to advance', 'line search failed'))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=1) + '\n')
    temporary.replace(path)


def provenance(config):
    return {
        'config': config, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'gpu': torch.cuda.get_device_name() if DEVICE == 'cuda' else None,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'cuml-cu13', 'numpy', 'scikit-learn', LIBRARY]},
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for root in ['src/role_probe_training/probes', 'src/role_probe_training/harmony',
                                       'src/experiments/gptoss_role_probes/header_roles']
                          for p in sorted(Path(root).glob('*.py'))} | {
            'src/role_probe_training/training/probe_fit.py':
                hashlib.sha256(Path('src/role_probe_training/training/probe_fit.py').read_bytes()).hexdigest()},
    }


def read_documents(config):
    lines = Path(config['documents']).read_bytes().splitlines(keepends=True)[:config['n_documents']]
    digest = hashlib.sha256(b''.join(lines)).hexdigest()
    if digest != config['documents_sha256']:
        raise ValueError(f'Documents differ from the recorded C4/Dolma passages: {digest}')
    documents = [json.loads(line) for line in lines]
    if any(d['source'] not in config['sources'] for d in documents):
        raise ValueError('A document is not from the configured sources')
    return documents


def extract(config, output):
    from role_probe_training.probes.model import extract_states, load_model
    folder = output / 'artifacts' / 'activations'
    if (folder / 'tokens.npz').exists():
        print('Activations already cached', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    roles, layers = ROLE_SETS[config['roles']], config['layers']
    documents = read_documents(config)
    wraps = []
    for index, document in enumerate(documents):
        for role in roles:
            enc = render_document(role, document['text'], document_header(document, role)).encoded(roles)
            positions = np.flatnonzero(enc['labels'] >= 0)
            wraps.append((index, role, enc['input_ids'], positions, enc['labels'][positions]))
    total = sum(len(wrap[3]) for wrap in wraps)
    tokenizer, model = load_model(config['model_path'])
    arrays = {layer: np.lib.format.open_memmap(folder / f'layer_{layer}.npy', mode='w+', dtype=np.float16,
                                               shape=(total, model.config.hidden_size)) for layer in layers}
    cursor = 0
    for number, (index, role, ids, positions, labels) in enumerate(wraps):
        states = extract_states(model, ids, layers, positions)
        for layer in layers:
            arrays[layer][cursor:cursor + len(positions)] = states[layer].astype(np.float16)
        cursor += len(positions)
        if (number + 1) % (50 * len(roles)) == 0:
            print(f'Extracted {(number + 1) // len(roles)}/{len(documents)} documents', flush=True)
    for array in arrays.values():
        array.flush()
    write_json(folder / 'metadata.json', {
        **provenance(config), 'roles': list(roles), 'tokens': total, 'hidden_size': model.config.hidden_size,
        'activation': 'post_attention_layernorm_output_pre_mlp', 'layer_indexing': 'zero_based', 'storage': 'float16',
        'precision': 'MXFP4 checkpoint dequantized to BF16', 'attention': 'eager',
        'rendering': RenderConfig().metadata(), 'wraps': 'one single-message rendering per passage and role; no system message'})
    np.savez(folder / 'tokens.npz',
             labels=np.concatenate([wrap[4] for wrap in wraps]).astype(np.int8),
             train=np.concatenate([np.full(len(wrap[3]), documents[wrap[0]]['split'] == 'train') for wrap in wraps]),
             # One example per rendered message, as the AdamW fitter batches by example.
             groups=np.concatenate([np.full(len(wrap[3]), number, dtype=np.int32) for number, wrap in enumerate(wraps)]))
    print(f'Cached {total} tokens for layers {layers}', flush=True)


@contextlib.contextmanager
def captured_output(path):
    """Send everything written to stdout and stderr, also by C++ libraries, to a file."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as target:
        os.dup2(target.fileno(), 1); os.dup2(target.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            with contextlib.suppress(Exception):
                from cuml.internals import logger
                logger.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])


def parse_solver_log(text):
    """Per-iteration objective and gradient norm, and why the cuML solver stopped."""
    history = [{'iteration': int(k), 'objective': float(f), 'criterion': float(c), 'gradient_norm': float(g)}
               for k, f, c, g in ITERATION.findall(text)]
    reasons = [name for marker, name in TERMINATIONS if marker in text]
    return history, (reasons[-1] if reasons else 'unknown')


@torch.no_grad()
def objective_audit(states, labels, probe, C, chunk=32768):
    """Mean cross-entropy, cuML objective (plus ||W||^2 / (2 C n)) and its largest gradient
    entry, in double precision over all given tokens. Same quantities as probes/audit.py."""
    weight = torch.as_tensor(probe.weight, dtype=torch.float64, device=DEVICE)
    bias = torch.as_tensor(probe.bias, dtype=torch.float64, device=DEVICE)
    gradient_weight, gradient_bias, loss = torch.zeros_like(weight), torch.zeros_like(bias), 0.
    for lo in range(0, len(labels), chunk):
        x = torch.as_tensor(states[lo:lo + chunk], device=DEVICE).double()
        y = torch.as_tensor(labels[lo:lo + chunk], dtype=torch.long, device=DEVICE)
        log_probabilities = torch.log_softmax(x @ weight.T + bias, dim=-1)
        loss += float(-log_probabilities.gather(1, y[:, None]).sum())
        residual = log_probabilities.exp()
        residual.scatter_add_(1, y[:, None], -torch.ones_like(residual[:, :1]))
        gradient_weight += residual.T @ x
        gradient_bias += residual.sum(0)
    n = len(labels)
    gradient_weight = gradient_weight / n + weight / (C * n)
    return {'mean_cross_entropy': loss / n, 'objective': loss / n + float(weight.square().sum()) / (2 * C * n),
            'max_abs_gradient': float(max(gradient_weight.abs().max(), (gradient_bias / n).abs().max()))}


def split_metrics(probe, states, labels, roles, chunk=65536):
    probabilities = np.concatenate([probe.predict_proba(states[lo:lo + chunk].astype(np.float32))
                                    for lo in range(0, len(labels), chunk)])
    return metrics(labels, probabilities, roles)


def fit_cuml(train, labels, config, metadata, roles, layer, output):
    """One cuML fit with the configured tolerance; its own stop criteria decide when it ends."""
    settings = config['cuml']
    log = output / 'logs' / f'cuml_layer_{layer}_tol{settings["tol"]:.0e}.log'
    start = time.monotonic()
    with captured_output(log):
        probe = fit_probe(train, labels, roles=roles, C=config['C'], max_iter=settings['max_iter'], backend='cuml',
                          dtype=settings['dtype'], metadata=metadata,
                          solver_options={'tol': settings['tol'], 'lbfgs_memory': settings['lbfgs_memory'],
                                          'verbose': settings['verbose']})
    history, termination = parse_solver_log(log.read_text())
    converged = termination in ('gradient test', 'loss-change test')
    seconds = time.monotonic() - start
    print(f'  cuML tol {settings["tol"]:g}: {probe.metadata["iterations"]} iterations, stopped by {termination}, {seconds:.0f}s',
          flush=True)
    probe.metadata.update(converged=converged, termination=termination,
                          termination_note='Converged means: the cuML solver stopped on its own gradient or loss-change test, '
                          'not at the iteration limit or on a failed line search.')
    return probe, {'converged': converged, 'tol': settings['tol'], 'iterations': probe.metadata['iterations'],
                   'termination': termination, 'seconds': seconds, 'history': history,
                   'history_quantity': 'cuML objective: mean cross-entropy + ||W||^2 / (2 C n), per L-BFGS iteration'}


def fit_adamw(train, labels, groups, validation, validation_labels, config, metadata, roles, earlier=()):
    """Lengthen the fixed schedule until the final training loss stops changing.

    Every schedule length is a fresh fit, so `earlier` runs of shorter schedules can be
    continued with longer ones in a later job.
    """
    settings, runs = config['adamw'], list(earlier)
    previous = runs[-1]['final_train_loss'] if runs else None
    held = (torch.as_tensor(train, device=DEVICE), torch.as_tensor(labels, dtype=torch.long, device=DEVICE),
            torch.as_tensor(validation, device=DEVICE), torch.as_tensor(validation_labels, dtype=torch.long, device=DEVICE))

    @torch.no_grad()
    def loss_of(probe, x, y, chunk=131072):
        total = sum(float(torch.nn.functional.cross_entropy(probe(x[lo:lo + chunk].float()).float(), y[lo:lo + chunk],
                                                            reduction='sum')) for lo in range(0, len(y), chunk))
        return total / len(y)

    for steps in [s for s in settings['steps'] if not runs or s > runs[-1]['steps']]:
        # Evaluate at steps spaced evenly on a log scale, which is how the loss curve is read.
        marks = set(np.unique(np.geomspace(1, steps, settings['evaluations']).round().astype(int)).tolist())
        history, start = [], time.monotonic()

        def on_step(done, batch_loss, probe):
            if done in marks:
                history.append({'step': done, 'batch_loss': batch_loss, 'train_loss': loss_of(probe, held[0], held[1]),
                                'validation_loss': loss_of(probe, held[2], held[3])})

        probe = fit_probe(train, labels, roles=roles, backend='adamw', max_iter=steps, groups=groups,
                          seed=config['seed'], metadata=metadata, on_step=on_step,
                          solver_options={k: settings[k] for k in ('learning_rate',) if k in settings})
        final = objective_audit(train, labels, probe, config['C'])['mean_cross_entropy']
        change = None if previous is None else abs(previous - final) / final
        converged = change is not None and change < settings['relative_tolerance']
        runs.append({'steps': steps, 'optimizer_steps': probe.metadata['fit']['optimizer_steps'], 'final_train_loss': final,
                     'relative_change_from_previous': change, 'converged': converged,
                     'seconds': time.monotonic() - start, 'history': history})
        print(f'  AdamW {steps} steps: final training loss {final:.6f}, relative change '
              f'{"n/a" if change is None else f"{change:.2e}"}, {runs[-1]["seconds"]:.0f}s', flush=True)
        previous = final
        if converged:
            break
    del held
    torch.cuda.empty_cache()
    probe.metadata.update(converged=converged, relative_change_from_previous=change,
                          termination_note='Converged means: the final training loss changed by less than relative_tolerance '
                          'when the number of steps was multiplied by the next factor in the schedule.')
    return probe, {'converged': converged, 'runs': runs,
                   'history_quantity': 'mean cross-entropy over all training (validation) tokens, in float32'}


def fit(config, output, layers, backends):
    cache = output / 'artifacts' / 'activations'
    tokens = np.load(cache / 'tokens.npz')
    cached = json.loads((cache / 'metadata.json').read_text())
    extraction = lambda c: {k: v for k, v in c.items() if k not in ('cuml', 'adamw')}  # fit settings may change
    if extraction(cached['config']) != extraction(config):
        raise ValueError('Cached activations were extracted for a different configuration')
    role_set = config['roles']
    roles = ROLE_SETS[role_set]
    labels, train_rows = tokens['labels'].astype(np.int64), tokens['train']
    validation_rows = ~train_rows
    fits, folders = output / 'metrics' / 'fits', {'cuml': 'probes_header', 'adamw': 'probes_header_adamw'}
    run = provenance(config)
    # Several fit processes can share one job, so the process ID is part of the name.
    write_json(output / 'metrics' / f'run_metadata_fit_{os.environ.get("SLURM_JOB_ID", "local")}_{os.getpid()}.json',
               {**run, 'activations': {k: cached[k] for k in ('git_commit', 'git_dirty', 'slurm_job_id', 'tokens', 'rendering')}})
    # An AdamW fit with another learning rate, or a cuML fit with another tolerance, is a variant
    # kept beside the fit with the default setting.
    rate, tol = config['adamw'].get('learning_rate'), config['cuml'].get('variant_tol')
    variant = {'cuml': 'cuml' if tol is None else f'cuml_tol{tol:.0e}', 'adamw': 'adamw' if rate is None else f'adamw_lr{rate:g}'}
    folders['adamw'] += '' if rate is None else f'_lr{rate:g}'
    folders['cuml'] += '' if tol is None else f'_cuml_tol{tol:.0e}'
    if tol is not None:
        config = {**config, 'cuml': {**config['cuml'], 'tol': tol}}

    def unfinished(layer, backend):
        """Not fitted yet, or an AdamW fit that has not converged and has longer schedules left to try."""
        path = fits / f'layer_{layer}_{variant[backend]}.json'
        if not path.exists():
            return True
        record = json.loads(path.read_text())
        return backend == 'adamw' and not record['converged'] and record['runs'][-1]['steps'] < config['adamw']['steps'][-1]

    for layer in layers:
        pending = [b for b in backends if unfinished(layer, b)]
        if not pending:
            continue
        states = np.load(cache / f'layer_{layer}.npy', mmap_mode='r')
        train, validation = np.ascontiguousarray(states[train_rows]), np.ascontiguousarray(states[validation_rows])
        train_labels, validation_labels = labels[train_rows], labels[validation_rows]
        metadata = {**{k: run[k] for k in ('git_commit', 'git_dirty', 'slurm_job_id', 'gpu', 'versions', 'source_sha256')},
                    'model_id': config['model_id'], 'model_revision': config['model_revision'], 'role_set': role_set,
                    'layer': layer, 'activation': cached['activation'], 'layer_indexing': cached['layer_indexing'],
                    'hidden_size': cached['hidden_size'], 'precision': cached['precision'], 'attention': cached['attention'],
                    'rendering': cached['rendering'], 'training_config': config, 'input_sha256': config['documents_sha256']}
        for backend in pending:
            print(f'Layer {layer}, {backend}, {role_set} roles: {len(train_labels)} training / '
                  f'{len(validation_labels)} validation tokens', flush=True)
            if backend == 'cuml':
                probe, record = fit_cuml(train, train_labels, config, metadata, roles, layer, output)
            else:
                path = fits / f'layer_{layer}_{variant[backend]}.json'
                earlier = json.loads(path.read_text())['runs'] if path.exists() else ()
                probe, record = fit_adamw(train, train_labels, tokens['groups'][train_rows], validation, validation_labels,
                                          config, metadata, roles, earlier)
            probe.save(output / folders[backend] / f'layer_{layer}.npz')
            probe = RoleProbe.load(output / folders[backend] / f'layer_{layer}.npz')
            record.update(layer=layer, backend=backend, variant=variant[backend], role_set=role_set, roles=list(roles),
                          learning_rate=probe.metadata.get('fit', {}).get('learning_rate'),
                          train=split_metrics(probe, train, train_labels, roles),
                          validation=split_metrics(probe, validation, validation_labels, roles),
                          # The cuML objective at this solution, whichever fitter produced it.
                          objective_audit=objective_audit(train, train_labels, probe, config['C']))
            write_json(fits / f'layer_{layer}_{variant[backend]}.json', record)
            print(f'Layer {layer}, {variant[backend]}: converged {record["converged"]}, training accuracy '
                  f'{record["train"]["accuracy"]:.4f}, validation accuracy {record["validation"]["accuracy"]:.4f}', flush=True)
        del train, validation


def compare(output, layers):
    """Agreement of every header probe of a layer with its cuML probe of the tightest tolerance.

    On the validation tokens: the share given the same role and the mean total-variation
    distance between the two probability vectors (half the summed absolute difference; 0 for
    equal probabilities, 1 for disjoint ones). For the weights: the cosine after subtracting
    the mean over roles from every role's weights, which leaves a softmax probe's
    probabilities unchanged.
    """
    cache = output / 'artifacts' / 'activations'
    held_out = ~np.load(cache / 'tokens.npz')['train']
    centred = lambda probe: (probe.weight - probe.weight.mean(0)).astype(np.float64).ravel()
    results = []
    for layer in layers:
        states = np.load(cache / f'layer_{layer}.npy', mmap_mode='r')[held_out].astype(np.float32)
        probes = {folder.name.removeprefix('probes_header').lstrip('_') or 'cuml': RoleProbe.load(folder / f'layer_{layer}.npz')
                  for folder in sorted(output.glob('probes_header*')) if (folder / f'layer_{layer}.npz').exists()}
        tightest = min((name for name in probes if name.startswith('cuml')),
                       key=lambda name: probes[name].metadata['solver_options']['tol'])
        reference = probes.pop(tightest)
        expected, weights = reference.predict_proba(states), centred(reference)
        for name, probe in probes.items():
            found, other = probe.predict_proba(states), centred(probe)
            results.append({'layer': layer, 'variant': name, 'reference': tightest, 'validation_tokens': int(len(states)),
                            'same_prediction': float((expected.argmax(1) == found.argmax(1)).mean()),
                            'mean_total_variation': float(0.5 * np.abs(expected - found).sum(1).mean()),
                            'weight_cosine': float(weights @ other / np.sqrt((weights @ weights) * (other @ other))),
                            'weight_norm_ratio': float(np.sqrt((other @ other) / (weights @ weights)))})
            print(f'Layer {layer}, {name} against {tightest}: same prediction {results[-1]["same_prediction"]:.4f}, '
                  f'total variation {results[-1]["mean_total_variation"]:.4f}, weight cosine {results[-1]["weight_cosine"]:.4f}', flush=True)
    write_json(output / 'metrics' / 'fitter_agreement.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['extract', 'fit', 'compare'])
    parser.add_argument('--config', default='src/experiments/gptoss_role_probes/header_roles/config.json')
    parser.add_argument('--output', type=Path, default=Path('experiment_reports/gptoss_role_probes/header_roles'))
    parser.add_argument('--layers', type=int, nargs='+', help='subset of the configured layers to fit')
    parser.add_argument('--backend', nargs='+', choices=['cuml', 'adamw'], default=['cuml', 'adamw'])
    parser.add_argument('--max-steps', type=int, help='try AdamW schedules up to this length only in this run')
    parser.add_argument('--learning-rate', type=float, help='AdamW peak learning rate other than upstream\'s 1e-3 (a separate variant)')
    parser.add_argument('--tol', type=float, help='cuML tolerance other than the configured one (a separate variant)')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.max_steps:
        config['adamw']['steps'] = [steps for steps in config['adamw']['steps'] if steps <= args.max_steps]
    if args.learning_rate:
        config['adamw']['learning_rate'] = args.learning_rate
    if args.tol and args.tol != config['cuml']['tol']:
        config['cuml']['variant_tol'] = args.tol
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    if args.stage == 'extract':
        extract(config, args.output)
    elif args.stage == 'compare':
        compare(args.output, args.layers or config['layers'])
    else:
        fit(config, args.output, args.layers or config['layers'], args.backend)
    print('Completed', args.stage, flush=True)


if __name__ == '__main__':
    main()
