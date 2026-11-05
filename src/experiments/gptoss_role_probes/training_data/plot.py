"""Held-out accuracy by layer for each training arm, one panel per test collection.

Colors are the first three categorical slots of the validated reference palette
(dataviz skill, references/palette.md), assigned to arms in fixed order.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = [('c4_dolma', 'C4/Dolma probe', '#2a78d6'), ('toucan', 'Toucan probe', '#eb6834'), ('combined', 'Combined probe', '#1baf7a')]
PANELS = [('neutral', 'Tested on C4/Dolma held-out'), ('toucan', 'Tested on Toucan held-out')]
INK, MUTED, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiment_reports/gptoss_role_probes/training_data'))
    args = parser.parse_args()
    results = {arm: json.loads((args.root / arm / 'metrics' / 'synthetic.json').read_text()) for arm, _, _ in ARMS}
    layers = sorted(int(L) for L in results['c4_dolma'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (collection, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.axhline(20, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
        ax.text(layers[0] - 0.8, 21.5, 'uniform / majority baseline', color=MUTED, fontsize=8, va='bottom')
        ends = []
        for arm, label, color in ARMS:
            y = [100 * results[arm][str(L)]['by_collection'][collection]['accuracy'] for L in layers]
            ax.plot(layers, y, color=color, linewidth=2, marker='o', markersize=6, markerfacecolor=SURFACE,
                    markeredgewidth=2, label=label)
            ends.append([y[-1], y[-1]])
        # Direct end labels, pushed apart vertically when they would overlap.
        for lower, upper in zip(sorted(ends), sorted(ends)[1:]):
            if upper[1] - lower[1] < 3.5:
                upper[1] = lower[1] + 3.5
        for value, position in ends:
            ax.annotate(f'{value:.1f}', (layers[-1], position), xytext=(6, 0), textcoords='offset points',
                        color=INK, fontsize=8, va='center')
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xticks(layers)
        ax.set_xlabel('Layer (zero-based, pre-MLP activations)', color=MUTED)
        ax.set_xlim(layers[0] - 1, layers[-1] + 3.5)
        ax.grid(axis='y', color='#e4e3df', linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#c8c7c2')
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel('Held-out token accuracy (%)', color=MUTED)
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 0.27), fontsize=9, labelcolor=INK)
    fig.suptitle('Role-probe accuracy by training data, GPT-OSS-20B', color=INK, fontsize=12, x=0.02, ha='left')
    fig.tight_layout()
    out = args.root / 'plots'
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'heldout_accuracy.png', dpi=180, facecolor=SURFACE)
    print('wrote', out / 'heldout_accuracy.png')


if __name__ == '__main__':
    main()
