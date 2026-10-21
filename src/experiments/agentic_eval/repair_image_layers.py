"""Recover checksum-verified official blobs absent from the public image cache."""
import hashlib
import json
from pathlib import Path
import requests

repairs=[('matplotlib__matplotlib-26342','swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-26342','079027039e9d5fa1677dd462c60fd9240097243a87ff8a124fffbd16f909c266'),('django__django-17084','swebench/sweb.eval.x86_64.django_1776_django-17084','180a31ca376f4176f9283f8acac2d286ba968dd251a3b279a89feedc5956584a')]
root=Path('.cache/apptainer-repair/cache/blob/blobs/sha256');root.mkdir(parents=True,exist_ok=True)
records=[]
for instance,repo,digest in repairs:
 target=root/digest
 if not target.exists():
  response=requests.get('https://auth.docker.io/token',params={'service':'registry.docker.io','scope':'repository:'+repo+':pull'},timeout=60);response.raise_for_status();token=response.json()['token']
  response=requests.get('https://registry-1.docker.io/v2/'+repo+'/blobs/sha256:'+digest,headers={'Authorization':'Bearer '+token},timeout=120,stream=True);response.raise_for_status()
  temporary=target.with_suffix('.tmp');h=hashlib.sha256()
  with temporary.open('wb') as out:
   for chunk in response.iter_content(1024*1024):out.write(chunk);h.update(chunk)
  if h.hexdigest()!=digest:raise ValueError('Blob checksum mismatch')
  temporary.replace(target)
 h=hashlib.sha256(target.read_bytes()).hexdigest();assert h==digest
 records.append({'sample_id':instance,'official_repository':repo,'sha256':digest,'bytes':target.stat().st_size,'cache_path':str(target)})
 print(instance,target.stat().st_size,flush=True)
Path('experiment_reports/agentic/swe_bench_full/image_layer_repairs.json').write_text(json.dumps(records,indent=2)+'\n')
