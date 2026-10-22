"""Supporting result tables; scientific narratives remain authored Markdown."""
from pathlib import Path


def write_probe_appendix(summary, output):
    roles = summary['roles']
    lines = [
        '# Confusion matrices', '',
        'Rows are ground-truth roles; columns are predicted roles. Counts include '
        'message-content tokens only. Headers, special tokens and synthetic tool-schema '
        'context are excluded. `cot` denotes assistant reasoning; `assistant` denotes '
        'non-reasoning assistant content, including tool calls. A missing true role '
        'has no recall estimate.', '',
        '## Supplementary metrics', '',
        '**Averaging.** Mean conversation accuracy gives each conversation equal weight. '
        'Role-balanced accuracy averages recall equally over roles present in the dataset; '
        'recall is the fraction of tokens of a true role classified correctly. Dolci has '
        'four observed roles; retail and SWE have five. Uniform random prediction over '
        'the five possible classes has expected accuracy 20%.', '',
        '**Segmentation.** Long histories are replayed in separate segments with context '
        'reset between them. “Segmented conversations” counts histories requiring more '
        'than one segment. “Unsegmented accuracy” is token accuracy restricted to histories '
        'that fit in one segment. It measures a different subset, not a controlled effect '
        'of segmentation. The segment size is documented in RUNNING.md.', '',
        '| Dataset | Layer | Content tokens | Mean conversation accuracy | Role-balanced accuracy | Segmented conversations | Unsegmented accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    def pct(value):
        return 'n/a' if value is None else f'{value:.2%}'
    for name, dataset in summary['datasets'].items():
        for layer, result in dataset['layers'].items():
            lines.append(f"| {name} | {layer} | {result['scored_tokens']:,} | "
                         f"{pct(result['macro_trajectory_accuracy'])} | "
                         f"{pct(result['balanced_role_accuracy'])} | "
                         f"{result['windowed_trajectories']} / {result['trajectories']} | "
                         f"{pct(result['single_window_accuracy'])} |")
    lines += ['', 'The [machine-readable results](metrics.json) also report mean true-role '
              'probability: the average probability assigned to the correct role, within '
              'each true role. This differs from accuracy, which uses the highest-probability '
              'class. Individual conversation results are included there and in the viewer.']
    for name, dataset in summary['datasets'].items():
        for layer, result in dataset['layers'].items():
            lines += ['', f'## {name} layer {layer}', '',
                      '| True / predicted | '+' | '.join(roles)+' |', '|---|'+'---:|'*len(roles)]
            for role, row in zip(roles, result['confusion_matrix']):
                lines.append('| '+role+' | '+' | '.join(map(str, row))+' |')
            lines += ['', 'Row-normalized percentages (the diagonal is role recall):', '',
                      '| True / predicted | '+' | '.join(roles)+' |', '|---|'+'---:|'*len(roles)]
            for role, row in zip(roles, result['confusion_matrix']):
                lines.append('| '+role+' | '+' | '.join(
                    f'{x/sum(row):.1%}' if sum(row) else 'n/a' for x in row)+' |')
    Path(output).write_text('\n'.join(lines)+'\n')
