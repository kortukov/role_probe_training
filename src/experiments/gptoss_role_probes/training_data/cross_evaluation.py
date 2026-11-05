"""One table per layer: each probe arm on its training set, both held-out sets, the hx2
AppWorld histories and No Robots version 2 (generated reasoning and answer)."""
import argparse
import json
from pathlib import Path

ARMS = [('c4_dolma', 'C4/Dolma'), ('toucan', 'Toucan'), ('combined', 'Combined')]
ROLES = ('system', 'user', 'cot', 'assistant', 'tool')


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x):
    return '—' if x is None else f'{100 * x:.1f}'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiment_reports/gptoss_role_probes/training_data'))
    parser.add_argument('--norobots', type=Path, default=Path('experiment_reports/eval_role_probes/chat_eval/norobots/metrics/version2_generated/metrics.json'))
    parser.add_argument('--layers', type=int, nargs='+', default=[12, 16])
    args = parser.parse_args()
    norobots = load(args.norobots)
    lines = ['# Cross-evaluation of the three probe sets\n',
             'Token accuracy in percent; macro recall is the mean of per-role recalls over roles present.',
             'Train and held-out sets are role-balanced (chance 20%). AppWorld is the hx2 base cohort',
             '(168 histories); No Robots is version 2 (489 first turns with generated reasoning and answer).\n']
    rows_json = {}
    for L in args.layers:
        lines.append(f'\n## Layer {L}\n')
        lines += ['| Probe | Train (own data) | Train C4/Dolma part | Train Toucan part | Held-out C4/Dolma | Held-out Toucan | AppWorld acc. | AppWorld macro | No Robots v2 acc. | No Robots v2 macro |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm, label in ARMS:
            syn = load(args.root / arm / 'metrics' / 'synthetic.json')[str(L)]
            audit = (load(args.root / arm / 'metrics' / 'optimizer_audit.json') or {}).get(str(L), {})
            aw = load(args.root / 'predictions' / 'appworld_test_normal' / arm / 'summary.json')
            aw = aw['layers'][str(L)] if aw else {}
            nr = norobots['arms'][arm][str(L)] if norobots else {}
            by = audit.get('train_accuracy_by_collection', {})
            row = {'train': audit.get('train_accuracy'), 'train_neutral': by.get('neutral'), 'train_toucan': by.get('toucan'),
                   'heldout_neutral': syn['by_collection']['neutral']['accuracy'], 'heldout_toucan': syn['by_collection']['toucan']['accuracy'],
                   'appworld_accuracy': aw.get('accuracy'), 'appworld_macro_recall': aw.get('macro_recall'),
                   'norobots_v2_accuracy': nr.get('accuracy'), 'norobots_v2_macro_recall': nr.get('macro_recall')}
            rows_json.setdefault(str(L), {})[arm] = row
            lines.append(f'| {label} | ' + ' | '.join(pct(row[k]) for k in row) + ' |')
        lines.append(f'\n**Per-role recall at layer {L}** (AppWorld / No Robots v2)\n')
        lines += ['| Probe | ' + ' | '.join(ROLES) + ' |', '|---|' + '---:|' * len(ROLES)]
        for arm, label in ARMS:
            aw = load(args.root / 'predictions' / 'appworld_test_normal' / arm / 'summary.json')
            aw_r = dict(zip(ROLES, aw['layers'][str(L)]['per_role_recall'])) if aw else {}
            nr_r = norobots['arms'][arm][str(L)]['per_role_recall'] if norobots else {}
            lines.append(f'| {label} | ' + ' | '.join(f'{pct(aw_r.get(r))} / {pct(nr_r.get(r))}' for r in ROLES) + ' |')
    (args.root / 'metrics' / 'cross_evaluation.md').write_text('\n'.join(lines) + '\n')
    (args.root / 'metrics' / 'cross_evaluation.json').write_text(json.dumps(rows_json, indent=1) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
