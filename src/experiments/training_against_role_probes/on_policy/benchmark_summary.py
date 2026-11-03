"""Summarize matched timings without turning failed runs into speedup claims."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    result = {'scope':'Serving benchmark on a fixed trained snapshot, not on-policy training',
              'limitations':['Weight-transfer time is not measured by this benchmark.',
                             'Backend numerical/policy equivalence is not established.',
                             'Episode runs use a diagnostic 300-second task cap.'], 'backends':{}}
    for index,name in [(0,'hf'),(1,'vllm')]:
        result['backends'][name]={}
        for phase in ['startup','warmup','micro','episodes']:
            path=args.root/str(index)/f'{phase}.json'
            result['backends'][name][phase]=json.loads(path.read_text()) if path.exists() else None
    hf,vl=result['backends']['hf'],result['backends']['vllm']
    result['same_adapter']=bool(hf['startup'] and vl['startup'] and
        hf['startup']['adapter_sha256']==vl['startup']['adapter_sha256'])
    if result['same_adapter'] and hf['micro'] and vl['micro']:
        result['micro_output_token_throughput_ratio_vllm_over_hf']={}
        for h,v in zip(hf['micro'],vl['micro'],strict=True):
            assert h['concurrency']==v['concurrency'] and len(h['requests'])==len(v['requests'])==4
            result['micro_output_token_throughput_ratio_vllm_over_hf'][h['concurrency']]=(
                v['output_tokens_per_second']/h['output_tokens_per_second'])
    result['complete']=all(result['backends'][b][p] is not None
        for b in ('hf','vllm') for p in ('startup','warmup','micro','episodes'))
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
