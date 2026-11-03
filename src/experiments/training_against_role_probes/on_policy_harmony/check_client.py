"""Verify malformed responses end Inspect's tool loop without a sample error."""
import json, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model, GenerateConfig
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool

@tool
def python():
    async def execute(code:str)->str:
        """Execute code.

        Args:
            code: Python source.
        """
        raise AssertionError('Malformed response must not invoke a tool')
    return execute

fixtures=[x for x in json.load(open('/tmp/onpolicy-parser-fixtures.json')) if x['invalid']]
assert len(fixtures)>=2
for fixture in fixtures:
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body=json.dumps(dict(id='audit',object='chat.completion',created=0,model='audit',choices=[dict(index=0,message=fixture['message'],finish_reason='stop')],usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2))).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=HTTPServer(('127.0.0.1',0),Handler); thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model=get_model('vllm/audit-'+fixture['case'],base_url=f'http://127.0.0.1:{server.server_port}/v1',api_key='local',stream=False,config=GenerateConfig(max_retries=0))
            logs=eval(Task(dataset=[Sample(input='Test')],solver=[use_tools(python()),generate(tool_calls='loop')]),model=model,log_dir=tmp,display='none')
            assert logs[0].status=='success' and not logs[0].samples[0].error, (fixture['case'], len(requests), logs[0].status, logs[0].samples[0].error.message if logs[0].samples[0].error else None)
            assert len(requests)==1
    finally:server.shutdown();thread.join();server.server_close()
print('Inspect ended each invalid-output episode after one response, with no tool invocation or sample error.')
