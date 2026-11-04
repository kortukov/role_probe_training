"""HF benchmark server with cancellation-aware serial generation.

Kept separate while queued on-policy jobs pin the original serving module.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import select
import socket
import threading
import time
import traceback
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from role_probe_training.on_policy.serving import parse_completion, template_messages


class ClientDisconnected(Exception):
    """The requester has gone away; this does not poison the model."""


def peer_disconnected(connection):
    ready, _, _ = select.select([connection], [], [], 0)
    if not ready:
        return False
    try:
        return connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
    except (ConnectionResetError, BrokenPipeError):
        return True
    except BlockingIOError:
        return False


class StopOnDisconnect(StoppingCriteria):
    def __init__(self, connection):
        self.connection = connection
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs):
        self.cancelled = self.cancelled or peer_disconnected(self.connection)
        return self.cancelled


@contextmanager
def serve(runtime, output, policy_step, generation, *, sampler=None):
    """Serve only while collecting; stop and join before any optimizer update."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = {'episode': None, 'request': 0, 'errors': [], 'disconnects': []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def disconnected(self, stage):
            if getattr(self, '_disconnect_recorded', False):
                return
            self._disconnect_recorded = True
            event = dict(episode_id=state['episode'], generation_index=state['request'],
                         stage=stage, time=time.time())
            state['disconnects'].append(event)
            with (output/'disconnects.jsonl').open('a') as stream:
                stream.write(json.dumps(event)+'\n')

        def reply(self, status, value):
            encoded = json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                self.disconnected('response_write')

        def do_GET(self):
            if self.path == '/v1/models':
                return self.reply(200, {'object': 'list', 'data': [{
                    'id': 'on-policy-role-model', 'object': 'model', 'created': 0,
                    'owned_by': 'local', 'max_model_len': generation['context_limit']}]})
            self.reply(404, {'error': 'Unknown route'})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/episode':
                    state['episode'] = body['id']
                    state['request'] = 0
                    return self.reply(200, {'ok': True})
                if self.path != '/v1/chat/completions' or state['episode'] is None:
                    raise ValueError('Unknown route or no active episode')
                if state['errors']:
                    return self.reply(409, {'error': {'message': 'Serving stopped after an earlier failure',
                        'type': 'on_policy_generation_error'}})
                if peer_disconnected(self.connection):
                    raise ClientDisconnected('before_generation')
                if body.get('stream'):
                    raise ValueError('Streaming is not supported')
                if body.get('temperature', 1) != 1 or body.get('top_p', 1) != 1:
                    raise ValueError('On-policy distribution must use temperature=1, top_p=1')
                prompt = runtime.tokenizer.apply_chat_template(
                    template_messages(body['messages']), tools=body.get('tools'),
                    tokenize=True, add_generation_prompt=True, reasoning_effort='high')
                maximum = generation['context_limit']
                budget = min(generation['max_tokens'], maximum - len(prompt))
                if budget <= 0:
                    raise ValueError('Full prompt reached serving context limit; no truncation')
                ids = torch.tensor([prompt], device=runtime.device)
                stops = [runtime.tokenizer.convert_tokens_to_ids(x) for x in ('<|return|>', '<|call|>')]
                seed = int(body['seed']) + state['request'] * 1000003
                # Record the exact failing input even if CUDA crashes before a
                # completion exists. These are diagnostics, not sampled data.
                with (output/'inputs.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(episode_id=state['episode'],
                        generation_index=state['request'], policy_step=policy_step,
                        seed=seed, prompt_ids=prompt))+'\n')
                started = time.perf_counter()
                cancellation = StopOnDisconnect(self.connection)
                if sampler is not None:
                    completion = sampler(prompt, budget, stops, seed)
                else:
                    with torch.random.fork_rng(devices=[runtime.device]), torch.no_grad():
                        torch.manual_seed(seed)
                        sampled = runtime.model.generate(
                            input_ids=ids, attention_mask=torch.ones_like(ids),
                            do_sample=True, temperature=1., top_p=1., top_k=0,
                            repetition_penalty=1., max_new_tokens=budget,
                            eos_token_id=stops, pad_token_id=runtime.pad_id, use_cache=True,
                            stopping_criteria=StoppingCriteriaList([cancellation]))
                    completion = sampled[0, len(prompt):].tolist()
                runtime.hidden = None
                row = dict(episode_id=state['episode'], policy_step=policy_step,
                           generation_index=state['request'], seed=seed,
                           prompt_ids=prompt, completion_ids=completion,
                           seconds=time.perf_counter()-started,
                           cancelled=cancellation.cancelled or peer_disconnected(self.connection))
                # Persist raw samples before parsing; malformed outputs are auditable.
                with (output/'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                if row['cancelled']:
                    self.disconnected('generation')
                    state['request'] += 1
                    return
                state['request'] += 1
                message = parse_completion(runtime.tokenizer.decode(completion, skip_special_tokens=False))
                for index, call in enumerate(message.get('tool_calls', [])):
                    call['id'] = f"{state['episode']}_{state['request']}_{index}"
                capped = completion[-1] not in stops
                reason = 'length' if capped else ('tool_calls' if message.get('tool_calls') else 'stop')
                return self.reply(200, {'id': f"{state['episode']}_{state['request']}",
                    'object': 'chat.completion', 'created': int(time.time()), 'model': body['model'],
                    'choices': [{'index': 0, 'message': message, 'finish_reason': reason}],
                    'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(completion),
                              'total_tokens': len(prompt)+len(completion)}})
            except (ClientDisconnected, BrokenPipeError, ConnectionResetError):
                runtime.hidden = None
                self.disconnected('request')
            except Exception as error:
                state['errors'].append(traceback.format_exc())
                (output/'server_errors.json').write_text(json.dumps(state['errors'], indent=2))
                self.reply(500, {'error': {'message': str(error), 'type': 'on_policy_generation_error'}})

    runtime.model.gradient_checkpointing_disable()
    server = HTTPServer(('127.0.0.1', 0), Handler)
    def serve_on_device():
        # CUDA current-device state is thread-local. The HTTP worker must bind
        # its rank's GPU explicitly before library code allocates on 'cuda'.
        if runtime.device != 'cpu':
            torch.cuda.set_device(runtime.device)
        server.serve_forever()

    thread = threading.Thread(target=serve_on_device, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        runtime.hidden = None
        runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
