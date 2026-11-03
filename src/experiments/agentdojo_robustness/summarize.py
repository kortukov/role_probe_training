"""Summarize every expected attempt, retaining missing/error coverage."""
import json
import tarfile
from pathlib import Path

ROOT = Path('experiment_reports/prompt_injection/agentdojo_same_batch_27000')
archive = ROOT / 'data/historical_base.tar.gz'
with tarfile.open(archive) as t:
    baseline = {m.name.split('/vllm_parsed/')[1]: json.load(t.extractfile(m)) for m in t.getmembers() if m.isfile() and m.name.endswith('.json')}
baseline = {k: v for k, v in baseline.items() if k.split('/')[1].startswith('user_task_') and not (k.startswith('workspace/') and k.split('/')[1] in {'user_task_35', 'user_task_38'})}

def summarize(records):
    out = {}
    for kind in ('clean', 'attacked'):
        keys = [k for k in baseline if ('/none/none.json' in k) == (kind == 'clean')]
        rows = [records.get(k, {}) for k in keys]
        skips = [bool(r.get('error')) and 'Skipping task' in str(r['error']) for r in rows]
        out[kind] = dict(expected=len(keys), present=sum(k in records for k in keys),
            scored=sum(r.get('utility') is not None for r in rows),
            utility=sum(r.get('utility') is True for r in rows)/len(keys),
            asr=sum(r.get('security') is True and not s for r,s in zip(rows,skips))/len(keys),
            asr_plus=sum(r.get('security') is True for r in rows)/len(keys),
            skips=sum(skips), errors=sum(bool(r.get('error')) for r in rows))
    return out

results = {'historical_base': summarize(baseline)}
for model in ('base', 'trained'):
    records = {}
    for f in (ROOT/'runs'/model/'vllm_parsed').glob('*/*/*/*.json'):
        records[str(f.relative_to(ROOT/'runs'/model/'vllm_parsed'))] = json.loads(f.read_text())
    for f in (ROOT/'runs'/model).glob('*.failures.jsonl'):
        for line in f.read_text().splitlines():
            failure = json.loads(line)
            if records.get(failure['key'], {}).get('utility') is None:
                records[failure['key']] = {**records.get(failure['key'], {}), 'error':failure['error']}
    results[model] = summarize(records)
(ROOT/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
lines = ['# AgentDojo results', '', 'Rates use all expected attempts; partial rows are provisional. ASR excludes harness skips from attack credit; ASR+ retains the recorded verdict. Missing attempts get no utility or attack credit; inspect coverage before comparing.', '', '| Model | Clean coverage | Clean utility | Attacked coverage | Attacked utility | ASR | ASR+ | Skips |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for model, r in results.items():
    c,a=r['clean'],r['attacked']
    if not c['present'] and not a['present']:
        lines.append(f"| {model} (pending) | 0/{c['expected']} | — | 0/{a['expected']} | — | — | — | — |")
        continue
    lines.append(f"| {model} | {c['scored']}/{c['expected']} | {c['utility']:.2%} | {a['scored']}/{a['expected']} | {a['utility']:.2%} | {a['asr']:.2%} | {a['asr_plus']:.2%} | {a['skips']} |")
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
