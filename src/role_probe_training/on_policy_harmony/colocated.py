"""Persistent vLLM process sharing a training GPU at synchronous barriers."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

import torch
from peft import get_peft_model_state_dict
from peft.tuners.lora.layer import ParamWrapper
from safetensors.torch import save_file


class ColocatedServer:
    def __init__(self, runtime, output):
        self.runtime = runtime
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.stage = Path(os.environ['TMPDIR'])/f'policy-transfer-rank{runtime.device}'
        self.stage.mkdir(parents=True, exist_ok=True)
        self.sleeping = False
        self.policy_step = None
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        project = Path('src/experiments/agentic_eval/server_current').resolve()
        env = os.environ.copy()
        devices = env.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
        env['CUDA_VISIBLE_DEVICES'] = devices[int(runtime.device)]
        for key in list(env):
            if key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                       'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT') or key.startswith('TORCHELASTIC_'):
                env.pop(key, None)
        env.update(VLLM_SERVER_DEV_MODE='1', VLLM_USE_FLASHINFER_SAMPLER='0',
                   CUDA_HOME=str(project/'.venv/lib/python3.12/site-packages/nvidia/cu13'))
        env['PATH'] = env['CUDA_HOME']+'/bin:'+env['PATH']
        env['TIKTOKEN_CACHE_DIR'] = str(Path('.cache/tiktoken').resolve())
        for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','VLLM_CACHE_ROOT'):
            env[key] = str(self.stage/key)
        model = Path('experiment_reports/training_against_role_probes/kl_iteration/same_batch/data/post_training/model').resolve()
        command = ['.tools/bin/uv','run','--project',str(project),'--frozen','--no-sync',
            'vllm','serve',str(model),'--served-model-name','on-policy-role-model',
            '--host','127.0.0.1','--port',str(port),'--max-model-len','32768',
            '--max-num-seqs','8','--gpu-memory-utilization','0.45',
            '--kv-cache-memory-bytes',str(8*1024**3),'--enable-sleep-mode',
            '--max-logprobs','-1',
            '--no-enable-prefix-caching','--generation-config','vllm',
            '--attention-backend','TRITON_ATTN','--moe-backend','triton',
            '--worker-cls','role_probe_training.on_policy_harmony.worker.HarmonyWorker',
            '--kernel-config','{"enable_jit_warmup":false}',
            '--cudagraph-capture-sizes','1','2','4','8','--max-cudagraph-capture-size','8']
        self.log = (self.output/'server.log').open('w')
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+900
            while time.monotonic()<deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'vLLM exited; see {self.output}/server.log')
                try:
                    with urllib.request.urlopen(self.url+'/health', timeout=2):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError('vLLM startup')
        except BaseException:
            self.close()
            raise

    def request(self, route, body=None):
        request = urllib.request.Request(self.url+route, data=json.dumps(body or {}).encode(),
                                        headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                data = response.read()
                return json.loads(data) if data else None
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    def sleep(self):
        self.request('/sleep?level=1&mode=wait')
        self.sleeping = True
        torch.cuda.empty_cache()

    @torch.no_grad()
    def synchronize(self, policy_step):
        started = time.monotonic()
        # Always stage while vLLM sleeps, leaving memory for merged expert tensors.
        if not self.sleeping:
            self.sleep()
        for index, layer in enumerate(self.runtime.base.model.layers):
            tensors = {}
            for module in layer.mlp.experts.modules():
                if isinstance(module, ParamWrapper):
                    value = module.get_param() + module.get_delta_weight('default')
                    tensors[f'model.layers.{index}.mlp.experts.{module.parameter_name}'] = value.cpu().contiguous()
            if len(tensors) != 2:
                raise ValueError('Expected both expert adapter projections')
            save_file(tensors, str(self.stage/f'experts{index}.safetensors'))
            del tensors
        attention = {name:value.detach().cpu().contiguous() for name,value in
                     get_peft_model_state_dict(self.runtime.model).items() if '.self_attn.' in name}
        if len(attention) != 192:
            raise ValueError('Attention adapter coverage changed')
        save_file(attention, str(self.stage/'attention.safetensors'))
        (self.stage/'manifest.json').write_text(json.dumps(dict(policy_step=policy_step)))
        torch.cuda.empty_cache()
        self.request('/wake_up?tags=weights')
        result = self.request('/collective_rpc', dict(method='refresh_role_policy',
                              args=[str(self.stage)], timeout=600))
        if result['results'] != [dict(policy_step=policy_step,verified_tensors=240)]:
            raise ValueError(f'Incomplete weight refresh: {result}')
        self.request('/wake_up?tags=kv_cache')
        self.sleeping = False
        self.policy_step = policy_step
        record = dict(policy_step=policy_step, seconds=time.monotonic()-started, **result)
        with (self.output/'refresh.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        return record

    def parse_completion(self, tokens, tools):
        response = self.request('/collective_rpc', dict(method='parse_role_completion', args=[tokens, tools], timeout=60))
        results = response['results']
        if len(results) != 1:
            raise ValueError('Expected one parser worker')
        return results[0]

    def sample(self, prompt, budget, stops, seed):
        if self.sleeping or self.policy_step is None:
            raise RuntimeError('Generation requires an awake synchronized policy')
        result = self.request('/v1/completions', dict(model='on-policy-role-model', prompt=prompt,
            max_tokens=budget, temperature=1., top_p=1., top_k=-1, seed=seed,
            stop_token_ids=stops, return_token_ids=True, skip_special_tokens=False))
        completion = result['choices'][0]['token_ids']
        if not completion or result['choices'][0].get('prompt_token_ids') != prompt:
            raise ValueError('Exact generation token IDs missing or changed')
        return completion

    @torch.no_grad()
    def compare_distribution(self, prompt):
        """Compare the complete next-token distribution on an actual context.

        The 0.1 nat KL bound is an engineering rejection threshold, not a claim
        of exact equivalence between HF and vLLM kernels.
        """
        ids = torch.tensor([prompt],device=self.runtime.device)
        hidden = self.runtime.base.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                                         use_cache=False).last_hidden_state
        log_q = self.runtime.base.lm_head(hidden[:,-1]).float().log_softmax(-1)[0].cpu().double()
        self.runtime.hidden = None
        del hidden
        result = self.request('/v1/completions',dict(model='on-policy-role-model',prompt=prompt,
            max_tokens=1,temperature=0,logprobs=-1,return_tokens_as_token_ids=True,
            return_token_ids=True))
        values = result['choices'][0]['logprobs']['top_logprobs'][0]
        log_p = torch.full_like(log_q,float('nan'))
        for name,value in values.items():
            if not name.startswith('token_id:'):
                raise ValueError('Token-ID log probabilities required')
            log_p[int(name.split(':')[1])] = value
        if not torch.isfinite(log_p).all():
            raise ValueError('Incomplete full-vocabulary probability comparison')
        kl = float((log_q.exp()*(log_q-log_p)).sum())
        result = dict(hf_to_vllm_kl=kl,hf_argmax=int(log_q.argmax()),
                      vllm_argmax=int(log_p.argmax()),vocabulary=len(log_q),
                      policy_step=self.policy_step)
        with (self.output/'distribution_checks.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
        if not -1e-5 <= kl < 0.1 or result['hf_argmax'] != result['vllm_argmax']:
            raise ValueError(f'Serving distribution gate failed: {result}')
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
