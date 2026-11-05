"""Tables for the No Robots probe evaluation (as-is replay and first-turn generation)."""
import argparse
import json
from pathlib import Path

ARM_LABELS = {'c4_dolma': 'C4/Dolma', 'toucan': 'Toucan', 'combined': 'Combined'}
ROLES = ('system', 'user', 'cot', 'assistant', 'tool')


def pct(x):
    return '—' if x is None else f'{100 * x:.1f}'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiment_reports/eval_role_probes/chat_eval/norobots'))
    args = parser.parse_args()
    versions = {}
    for name in ('version1_replay', 'version2_generated'):
        path = args.root / 'metrics' / name / 'metrics.json'
        if path.exists():
            versions[name] = json.loads(path.read_text())
    lines = []
    summary = json.loads((args.root / 'artifacts' / 'generation_summary.json').read_text()) if (args.root / 'artifacts' / 'generation_summary.json').exists() else None
    if summary:
        lines.append('### Version 2 generation\n')
        lines.append(f"| Prompts | Completed | Failed | Mean completion tokens | Mean reasoning characters (completed) |\n|---:|---:|---:|---:|---:|\n"
                     f"| {summary['n_prompts']} | {summary['completed']} | {summary['failed']} | {summary['mean_completion_tokens']:.0f} | {summary['mean_reasoning_characters_completed']:.0f} |")
        if summary['failure_reasons']:
            lines.append('\nFailure reasons: ' + '; '.join(f'{k} ({v})' for k, v in summary['failure_reasons'].items()))
    for name, data in versions.items():
        title = 'Version 1: conversations as given' if name.startswith('version1') else 'Version 2: first turn with generated reasoning and answer'
        arms = list(data['arms'])
        layers = sorted(int(L) for L in data['arms'][arms[0]])
        lines.append(f'\n### {title} ({data["n_conversations"]} conversations)\n')
        lines.append('Overall token accuracy / macro recall (%) by layer.\n')
        lines += ['| Arm | ' + ' | '.join(str(L) for L in layers) + ' |', '|---|' + '---:|' * len(layers)]
        for a in arms:
            lines.append(f'| {ARM_LABELS.get(a, a)} | ' + ' | '.join(f'{pct(data["arms"][a][str(L)]["accuracy"])} / {pct(data["arms"][a][str(L)]["macro_recall"])}' for L in layers) + ' |')
        for L in (12, 16):
            if L not in layers:
                continue
            roles = [r for r in ROLES if any(data['arms'][a][str(L)]['role_support'][r] for a in arms)]
            lines.append(f'\n**Per-role recall (%) at layer {L}**\n')
            lines += ['| Arm | ' + ' | '.join(roles) + ' |', '|---|' + '---:|' * len(roles)]
            for a in arms:
                lines.append(f'| {ARM_LABELS.get(a, a)} | ' + ' | '.join(pct(data['arms'][a][str(L)]['per_role_recall'][r]) for r in roles) + ' |')
            turns = list(data['arms'][arms[0]][str(L)]['per_turn'])
            def order(t):
                if t == 'system': return (0, 0)
                kind, k = t.split(); return (int(k), {'user': 0, 'reasoning': 1, 'assistant': 2}[kind])
            turns.sort(key=order)
            lines.append(f'\n**Per-turn token accuracy (%) at layer {L}** (tokens in the turn across conversations in parentheses)\n')
            lines += ['| Arm | ' + ' | '.join(turns) + ' |', '|---|' + '---:|' * len(turns)]
            for a in arms:
                pt = data['arms'][a][str(L)]['per_turn']
                lines.append(f'| {ARM_LABELS.get(a, a)} | ' + ' | '.join(f'{pct(pt[t]["accuracy"])} ({pt[t]["tokens"]})' if t in pt else '—' for t in turns) + ' |')
            if name.startswith('version2'):
                lines.append(f'\n**System-turn accuracy (%) at layer {L} by whether the sample supplied a system message**\n')
                lines += ['| Arm | With sample system text | Without |', '|---|---:|---:|']
                for a in arms:
                    s = data['arms'][a][str(L)]['system_turn_by_source']
                    lines.append(f'| {ARM_LABELS.get(a, a)} | {pct(s["with_sample_system"]["accuracy"])} ({s["with_sample_system"]["tokens"]}) | {pct(s["without_sample_system"]["accuracy"])} ({s["without_sample_system"]["tokens"]}) |')
    (args.root / 'metrics').mkdir(exist_ok=True)
    (args.root / 'metrics' / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
