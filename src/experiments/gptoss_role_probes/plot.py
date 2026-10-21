"""Regenerate report heatmaps from saved metrics, without loading the model."""

import argparse
import json
from pathlib import Path

from role_probe_training.probes.evaluation import plot_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiment_reports/role_probes/gptoss20b_baseline')
    args = parser.parse_args()
    root = Path(args.output)
    (root/'plots').mkdir(parents=True, exist_ok=True)
    data = json.loads((root/'metrics'/'conversations.json').read_text())
    for layer, result in data.items():
        plot_summary(result['conversation_balanced'], root/'plots'/f'clean_chat_layer_{layer}.png',
                     f'GPT-OSS-20B clean conversations — block {layer}')
    reference = root/'metrics'/'previous_probe_comparison.json'
    if reference.exists():
        plot_summary(json.loads(reference.read_text())['conversation_balanced'],
                     root/'plots'/'previous_probe_layer_12.png',
                     'Historical layer-12 probe on the same clean conversations')


if __name__ == '__main__':
    main()
