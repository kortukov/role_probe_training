"""Row-normalized confusion tables for the three probe sets on every evaluated set.

For each probe arm and layer: one overall accuracy table (train, both held-out
collections, AppWorld, No Robots v2), then one table per data set giving, for each true
role, the percentage of its tokens predicted as each of the five roles. Counts come from
the saved metrics: optimizer_audit.json (train), synthetic.json (held-out),
predictions/appworld_test_normal/<arm>/summary.json (AppWorld) and the No Robots
version-2 metrics.json.
"""
import argparse
import json
from pathlib import Path

ARMS = [('c4_dolma', 'C4/Dolma'), ('toucan', 'Toucan'), ('combined', 'Combined')]
ROLES = ('system', 'user', 'cot', 'assistant', 'tool')
COLUMNS = ('system', 'user', 'reasoning', 'assistant', 'tool')
SETS = [('train', 'Train (own fitting data)'), ('heldout_neutral', 'Held-out C4/Dolma'),
        ('heldout_toucan', 'Held-out Toucan'), ('appworld', 'AppWorld (168 histories)'),
        ('norobots_v2', 'No Robots v2 (489 generated first turns)')]


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def accuracy(counts):
    total = sum(map(sum, counts))
    return sum(counts[i][i] for i in range(len(ROLES))) / total if total else None


def pct(x):
    return '—' if x is None else f'{100 * x:.1f}'


def confusion_table(counts):
    lines = ['| True role (num tokens) | ' + ' | '.join(f'→ {c}' for c in COLUMNS) + ' |', '|---|' + '---:|' * len(COLUMNS)]
    for i, role in enumerate(COLUMNS):
        total = sum(counts[i])
        if total == 0:
            lines.append(f'| {role} (0) | ' + ' | '.join('—' for _ in COLUMNS) + ' |')
            continue
        cells = [f'{100 * counts[i][j] / total:.1f}' for j in range(len(COLUMNS))]
        cells[i] = f'**{cells[i]}**'
        lines.append(f'| {role} ({total:,}) | ' + ' | '.join(cells) + ' |')
    return lines


def gather(root, norobots, arm, layer):
    audit = (load(root / arm / 'metrics' / 'optimizer_audit.json') or {}).get(str(layer), {})
    syn = load(root / arm / 'metrics' / 'synthetic.json')[str(layer)]
    appworld = load(root / 'predictions' / 'appworld_test_normal' / arm / 'summary.json')
    counts = {
        'train': audit.get('train_confusion_counts'),
        'heldout_neutral': syn['by_collection']['neutral']['confusion_counts'],
        'heldout_toucan': syn['by_collection']['toucan']['confusion_counts'],
        'appworld': appworld['layers'][str(layer)]['confusion_counts'] if appworld else None,
        'norobots_v2': norobots['arms'][arm][str(layer)]['confusion_counts'] if norobots else None,
    }
    for collection, matrix in (audit.get('train_confusion_counts_by_collection') or {}).items():
        counts[f'train_{collection}'] = matrix
    return counts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiment_reports/gptoss_role_probes/training_data'))
    parser.add_argument('--norobots', type=Path, default=Path('experiment_reports/eval_role_probes/chat_eval/norobots/metrics/version2_generated/metrics.json'))
    parser.add_argument('--layers', type=int, nargs='+', default=[12, 16])
    args = parser.parse_args()
    norobots = load(args.norobots)
    lines = ['# Confusion tables for the three probe sets\n',
             'Each confusion table is row-normalized: for every true role, the percentage of its tokens',
             'predicted as each role (rows sum to 100; the diagonal is the per-role recall). Token counts',
             'per true role are in parentheses. Train and held-out sets wrap every document in all five',
             'roles; AppWorld and No Robots carry their natural role mix, and No Robots has no tool tokens.\n']
    out = {}
    for layer in args.layers:
        lines.append(f'\n## Layer {layer}\n')
        gathered = {arm: gather(args.root, norobots, arm, layer) for arm, _ in ARMS}
        out[str(layer)] = gathered
        lines.append('**Overall token accuracy (%)**\n')
        lines += ['| Probe | ' + ' | '.join(name for _, name in SETS) + ' |', '|---|' + '---:|' * len(SETS)]
        for arm, label in ARMS:
            lines.append(f'| {label} | ' + ' | '.join(pct(accuracy(m)) if (m := gathered[arm].get(key)) else '—' for key, _ in SETS) + ' |')
        lines.append('| Num tokens | ' + ' | '.join('per probe, see below' if key == 'train' else f'{sum(map(sum, m)):,}' if (m := gathered[ARMS[0][0]].get(key)) else '—' for key, _ in SETS) + ' |')
        lines.append('\nTrain token counts differ per probe: ' + ', '.join(f'{label} {sum(map(sum, gathered[arm]["train"])):,}' for arm, label in ARMS if gathered[arm].get('train')) + '.')
        for arm, label in ARMS:
            counts = gathered[arm]
            lines.append(f'\n### {label} probe, layer {layer}\n')
            for key, name in SETS:
                matrix = counts.get(key)
                lines.append(f'\n**{name}**\n')
                lines += confusion_table(matrix) if matrix else ['_not available_']
            extra = [k for k in counts if k.startswith('train_') and counts[k]]
            if len(extra) > 1:
                for key in extra:
                    lines.append(f'\n**Train, {key.removeprefix("train_")} part only**\n')
                    lines += confusion_table(counts[key])
    (args.root / 'metrics' / 'confusion_tables.md').write_text('\n'.join(lines) + '\n')
    (args.root / 'metrics' / 'confusion_tables.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
