"""Real socket regression checks: cancellation, write races and fatal failures."""
from contextlib import nullcontext
import http.client
import json
from pathlib import Path
import socket
import socketserver
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request
import urllib.error
import torch
from transformers import AutoTokenizer
from role_probe_training.generation.serving import serve


def main():
    tokenizer=AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b',local_files_only=True)
    tokens=tokenizer.encode('<|channel|>final<|message|>Two.<|return|>',add_special_tokens=False)
    started=threading.Event();stopped=threading.Event()
    class Model:
        calls=0
        slow=True
        fail=False
        def gradient_checkpointing_disable(self):pass
        def gradient_checkpointing_enable(self,**kwargs):pass
        def generate(self,input_ids,stopping_criteria,**kwargs):
            self.calls+=1
            if self.fail:raise RuntimeError('deliberate model failure')
            if self.slow:
                started.set()
                for _ in range(500):
                    if stopping_criteria(input_ids,None).all():
                        stopped.set();break
                    time.sleep(.01)
                else:raise AssertionError('Disconnected generation did not stop')
            return torch.cat([input_ids,torch.tensor([tokens])],dim=1)
    model=Model()
    runtime=SimpleNamespace(model=model,tokenizer=tokenizer,device='cpu',pad_id=tokenizer.pad_token_id,hidden=None)
    body=dict(model='on-policy-role-model',messages=[dict(role='user',content='One plus one?')],seed=123)
    def post(url,value):
        request=urllib.request.Request(url,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)
    with TemporaryDirectory() as directory,patch('torch.random.fork_rng',return_value=nullcontext()):
        with serve(runtime,directory,0,dict(max_tokens=128,context_limit=32768)) as (url,state):
            post(url+'/episode',dict(id='test'))
            port=int(url.rsplit(':',1)[1])
            connection=socket.create_connection(('127.0.0.1',port))
            payload=json.dumps(body).encode()
            connection.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '+str(len(payload)).encode()+b'\r\n\r\n'+payload)
            assert started.wait(3)
            connection.close()
            assert stopped.wait(2),'GPU worker equivalent remained busy after disconnect'
            model.slow=False
            assert post(url+'/v1/chat/completions',body)['choices'][0]['message']['content']=='Two.'
            assert not state['errors'] and len(state['disconnects'])==1
            rows=[json.loads(s) for s in (Path(directory)/'requests.jsonl').read_text().splitlines()]
            assert rows[0]['cancelled'] and not rows[1]['cancelled']
            # Simulate the precise observed race: disconnect only during write.
            original=socketserver._SocketWriter.write
            injected=[]
            def broken_write(writer,data):
                if not injected:
                    injected.append(True);raise BrokenPipeError('client closed before reply')
                return original(writer,data)
            with patch.object(socketserver._SocketWriter,'write',broken_write):
                try:post(url+'/v1/chat/completions',body)
                except (http.client.RemoteDisconnected,ConnectionResetError,urllib.error.URLError):pass
                else:raise AssertionError('Expected disconnected HTTP response')
            assert post(url+'/v1/chat/completions',body)['choices'][0]['message']['content']=='Two.'
            assert len(state['disconnects'])==2 and not state['errors']
            # Transport handling must not hide model/CUDA failures.
            model.fail=True
            for expected in (500,409):
                try:post(url+'/v1/chat/completions',body)
                except urllib.error.HTTPError as error:assert error.code==expected
                else:raise AssertionError('Model failure must stop serving')
            assert len(state['errors'])==1
    print('PASS: active cancellation, partial-token audit, response-write disconnect, later requests, and fatal-model guard')

if __name__=='__main__':main()
