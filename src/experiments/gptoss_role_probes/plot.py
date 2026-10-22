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


if __name__ == '__main__':
    main()
