"""Audit existing SWE attempts without regenerating or changing selected cohorts."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('experiment_reports/training_against_role_probes')
SWE = Path('experiment_reports/agentic/swe_bench_full')


def read(path):
    return [json.loads(line) for line in path.open()]


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts['tasks'] += 1
        counts['evaluation_errors'] += bool(row['error'])
        counts['execution_limit_events'] += bool(row['limit_events'])
        counts['histories_with_output_cap'] += bool(row['generation_token_caps'])
        context = any("exceeds model's maximum context length" in str(m.get('content', ''))
                      or 'maximum context length is' in str(m.get('content', ''))
                      for m in row['messages'] if m['role'] == 'assistant')
        counts['context_limit_responses'] += context
        score = row['scores'].get('swe_bench_scorer', {})
        counts['solved'] += score.get('value') == 1
        counts['solved_despite_context_limit'] += context and score.get('value') == 1
        if score:
            counts['scored'] += 1
            empty = not score.get('metadata', {}).get('model_patch', '').strip()
            counts['empty_scored_patches'] += empty
            counts['empty_scored_patches_with_context_limit'] += empty and context
    return dict(counts)


paths = {'base_original': [SWE/'data/trajectories.jsonl'],
         'base_repaired_partial': [SWE/f'runs/{run}/trajectories.jsonl'
                                   for run in ['1492764', '1492765']]}
paths.update({label: [ROOT/sub/'data/post_training/swe_bench/trajectories.jsonl']
              for label, sub in [('AppWorld-trained', ''), ('ASTRA-trained', 'astra_sft_1k'),
                                 ('DepthBench-trained', 'depthbench')]})
results = {}
cohorts = {}
for label, sources in paths.items():
    rows = [row for source in sources for row in read(source)]
    assert len({row['sample_id'] for row in rows}) == len(rows)
    cohorts[label] = rows
    results[label] = {'counts': summarize(rows), 'sources': [
        {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
ids = {row['sample_id'] for row in cohorts['base_repaired_partial']}
results['base_original_on_repaired_task_ids'] = {'counts': summarize(
    [row for row in cohorts['base_original'] if row['sample_id'] in ids])}
(ROOT/'post_training/swe_failure_audit.json').write_text(json.dumps(results, indent=2)+'\n')
for label, result in results.items():
    print(label, result['counts'])
