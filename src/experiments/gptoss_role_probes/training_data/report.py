"""Assemble the probe training-data comparison tables from saved arm metrics.

Reads each arm's synthetic.json, conversations.json, optimizer_audit.json and probe
metadata, plus optional whole-history AppWorld summaries, and writes
metrics/summary.json and metrics/tables.md under the experiment root.
"""
import argparse
import json
from pathlib import Path

from role_probe_training.harmony import ROLES

ARM_LABELS = {'c4_dolma': 'C4/Dolma', 'toucan': 'Toucan', 'combined': 'Combined'}
COLLECTIONS = {'neutral': 'C4/Dolma held-out', 'toucan': 'Toucan held-out'}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x):
    return f'{100 * x:.1f}'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiment_reports/gptoss_role_probes/training_data'))
    parser.add_argument('--arms', nargs='+', default=['c4_dolma', 'toucan', 'combined'])
    parser.add_argument('--appworld', type=Path, help='Whole-history prediction root with one folder per arm')
    args = parser.parse_args()
    summary, lines = {'arms': {}}, []
    arms = {}
    for arm in args.arms:
        d = args.root / arm
        arms[arm] = {'synthetic': load(d / 'metrics' / 'synthetic.json'),
                     'conversations': load(d / 'metrics' / 'conversations.json'),
                     'audit': load(d / 'metrics' / 'optimizer_audit.json'),
                     'probes': {L: load(d / 'probes' / f'layer_{L}.json') for L in (4, 8, 12, 16, 20)},
                     'appworld': load(args.appworld / arm / 'summary.json') if args.appworld else None}
        summary['arms'][arm] = {k: v for k, v in arms[arm].items() if k != 'probes'}
        summary['arms'][arm]['fit'] = {str(L): {k: m.get(k) for k in ('iterations', 'converged', 'warnings')}
                                       for L, m in arms[arm]['probes'].items() if m}
    layers = sorted({int(L) for a in arms.values() if a['synthetic'] for L in a['synthetic']})
    present = [a for a in args.arms if arms[a]['synthetic']]

    lines.append('### Held-out token accuracy (%) by training arm and test collection\n')
    lines.append('Uniform and majority baselines are 20.0 in every cell.\n')
    header = '| Layer | ' + ' | '.join(f'{ARM_LABELS[a]} → {COLLECTIONS[c]}' for a in present for c in COLLECTIONS) + ' |'
    lines += [header, '|---|' + '---:|' * (len(present) * len(COLLECTIONS))]
    for L in layers:
        row = [str(L)]
        for a in present:
            r = arms[a]['synthetic'].get(str(L))
            for c in COLLECTIONS:
                row.append(pct(r['by_collection'][c]['accuracy']) if r else 'pending')
        lines.append('| ' + ' | '.join(row) + ' |')

    for L in (12, 16):
        lines.append(f'\n### Per-role recall (%) at layer {L}\n')
        lines += ['| Arm | Test collection | ' + ' | '.join(ROLES) + ' |', '|---|---|' + '---:|' * len(ROLES)]
        for a in present:
            r = arms[a]['synthetic'].get(str(L))
            if not r:
                continue
            for c, name in COLLECTIONS.items():
                by = r['by_collection'][c]['by_role']
                lines.append(f'| {ARM_LABELS[a]} | {name} | ' + ' | '.join(pct(by[role]['accuracy']) for role in ROLES) + ' |')

    lines.append('\n### Clean chat: conversation-balanced mean true-role probability (%), layer 12 / 16\n')
    lines += ['| Arm | Role | Layer 12 | Layer 16 | Conversations |', '|---|---|---:|---:|---:|']
    for a in present:
        conv = arms[a]['conversations']
        if not conv:
            lines.append(f'| {ARM_LABELS[a]} | pending | | | |')
            continue
        for role in ROLES:
            cells = []
            n = None
            for L in ('12', '16'):
                s = conv.get(L, {}).get('conversation_balanced', {}).get(role)
                cells.append(pct(s['mean_probabilities'][ROLES.index(role)]) if s else '—')
                n = s['n_conversations'] if s else n
            if n is not None:
                lines.append(f'| {ARM_LABELS[a]} | {role} | {cells[0]} | {cells[1]} | {n} |')

    if args.appworld:
        lines.append('\n### AppWorld normal test, whole histories: overall accuracy and per-role recall in percent\n')
        lines += ['| Arm | Layer | Overall | Macro recall | ' + ' | '.join(ROLES) + ' | Scored |', '|---|---:|---:|---:|' + '---:|' * len(ROLES) + '---:|']
        for a in present:
            aw = arms[a]['appworld']
            if not aw:
                lines.append(f'| {ARM_LABELS[a]} | pending | | | | | | | | |')
                continue
            for L, r in sorted(aw['layers'].items(), key=lambda kv: int(kv[0])):
                lines.append(f'| {ARM_LABELS[a]} | {L} | {pct(r["accuracy"])} | {pct(r["macro_recall"])} | ' +
                             ' | '.join(pct(x) for x in r['per_role_recall']) + f' | {aw["scored_trajectories"]}/{aw["trajectories"]} |')

    lines.append('\n### Fit diagnostics\n')
    lines += ['| Arm | Layer | Train tokens | L-BFGS iterations | Audit max abs gradient |', '|---|---:|---:|---:|---:|']
    for a in present:
        for L in layers:
            m = arms[a]['probes'].get(L)
            au = (arms[a]['audit'] or {}).get(str(L))
            if m:
                lines.append(f'| {ARM_LABELS[a]} | {L} | {au["n_train_tokens"] if au else "—"} | {m["iterations"]} | '
                             f'{au["max_abs_gradient"]:.1e}' if au else f'| {ARM_LABELS[a]} | {L} | — | {m["iterations"]} | pending')
                if au:
                    lines[-1] += ' |'
                else:
                    lines[-1] += ' |'
    appendix = ['# Appendix: confusion matrices and diagnostics\n',
                'Rows are true roles, columns predicted roles, as percentages of each true role\'s',
                'held-out tokens. Both collections are role-balanced by construction.\n']
    for a in present:
        for c, name in COLLECTIONS.items():
            appendix.append(f'\n## {ARM_LABELS[a]} probe, {name}\n')
            for L in layers:
                r = arms[a]['synthetic'].get(str(L))
                if not r:
                    continue
                counts = r['by_collection'][c]['confusion_counts']
                appendix.append(f'\n**Layer {L}** (accuracy {pct(r["by_collection"][c]["accuracy"])}%)\n')
                appendix += ['| True \\ Predicted | ' + ' | '.join(ROLES) + ' |', '|---|' + '---:|' * len(ROLES)]
                for i, role in enumerate(ROLES):
                    total = sum(counts[i]) or 1
                    appendix.append(f'| {role} | ' + ' | '.join(f'{100 * v / total:.1f}' for v in counts[i]) + ' |')
    appendix.append('\n## Clean chat token accuracy by layer (%)\n')
    appendix += ['| Arm | ' + ' | '.join(str(L) for L in layers) + ' |', '|---|' + '---:|' * len(layers)]
    for a in present:
        conv = arms[a]['conversations']
        if conv:
            appendix.append(f'| {ARM_LABELS[a]} | ' + ' | '.join(pct(conv[str(L)]['token_metrics']['accuracy']) if str(L) in conv else '—' for L in layers) + ' |')
    if args.appworld:
        appendix.append('\n## AppWorld normal test: percentage confusion matrices\n')
        for a in present:
            aw = arms[a]['appworld']
            if not aw:
                continue
            for L, r in sorted(aw['layers'].items(), key=lambda kv: int(kv[0])):
                appendix.append(f'\n**{ARM_LABELS[a]} probe, layer {L}** (accuracy {pct(r["accuracy"])}%, macro recall {pct(r["macro_recall"])}%)\n')
                appendix += ['| True \\ Predicted | ' + ' | '.join(ROLES) + ' |', '|---|' + '---:|' * len(ROLES)]
                for i, role in enumerate(ROLES):
                    total = sum(r['confusion_counts'][i]) or 1
                    appendix.append(f'| {role} | ' + ' | '.join(f'{100 * v / total:.1f}' for v in r['confusion_counts'][i]) + ' |')
    (args.root / 'APPENDIX.md').write_text('\n'.join(appendix) + '\n')
    (args.root / 'metrics').mkdir(exist_ok=True)
    (args.root / 'metrics' / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (args.root / 'metrics' / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
