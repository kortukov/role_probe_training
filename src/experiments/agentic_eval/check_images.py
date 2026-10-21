"""Resolve official SWE images through Google's public Docker Hub cache."""
import concurrent.futures
import json
from pathlib import Path
import re
import requests

root=Path('experiment_reports/agentic/swe_bench_full')
ids=json.loads((root/'preparation/manifest.json').read_text())['full_selection']
def check(instance):
 m=re.fullmatch(r'(.+?)__(.+)-(\d+)',instance); org,repo,issue=m.groups()
 name=f'swebench/sweb.eval.x86_64.{org}_1776_{repo}-{issue}'
 response=requests.head('https://mirror.gcr.io/v2/'+name+'/manifests/latest',headers={'Accept':'application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json'},timeout=60)
 digest=response.headers.get('docker-content-digest')
 return instance,{'source':'docker.io/'+name+':latest','mirror':'mirror.gcr.io/'+name,'status':response.status_code,'digest':digest,'image':'mirror.gcr.io/'+name+'@'+digest if digest else None}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: result=dict(pool.map(check,ids))
(root/'image_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
from collections import Counter
print(Counter(v['status'] for v in result.values()),sum(v['image'] is not None for v in result.values()))
