"""Serve the colocated vLLM policy to concurrent episodes at synchronous barriers.

Each episode uses its own route, /episodes/<id>/v1, so generation indices, seeds,
tool-call IDs and retained raw completions are per episode. vLLM batches the
generations. Inspect runs in its existing separate environment.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
import traceback


ROUTE = re.compile(r'/episodes/([^/]+)/v1/(chat/completions|models)')


class Server(ThreadingHTTPServer):
    request_queue_size = 128


def template_messages(messages, raw_completions=None):
    result = []
    for original in messages:
        message = dict(original)
        reasoning = message.pop('reasoning_content', None)
        if reasoning:
            message['thinking'] = reasoning
        if message.get('tool_calls'):
            # Inspect inserts a newline for an otherwise empty assistant text
            # block. GPT-OSS treats any nonempty content as a second reasoning
            # field alongside thinking and rejects it. Only remove whitespace;
            # never discard meaningful assistant text to satisfy the template.
            if message.get('thinking') and not (message.get('content') or '').strip():
                message['content'] = ''
            message['tool_calls'] = json.loads(json.dumps(message['tool_calls']))
            for call in message['tool_calls']:
                arguments = call['function']['arguments']
                if isinstance(arguments, str):
                    call['function']['arguments'] = json.loads(arguments)
        if message.get('tool_calls') and ((message.get('thinking') and message.get('content')) or '<|channel|>' in str(message.get('thinking', '')) or '<|channel|>' in str(message.get('content', ''))):
            ids = tuple(call['id'] for call in message['tool_calls'])
            if raw_completions is None or ids not in raw_completions:
                raise ValueError('Mixed-channel tool message has no retained raw completion')
            message['raw_harmony'] = raw_completions[ids]
        result.append(message)
    return result


@contextmanager
def serve(runtime, output, policy_step, generation, *, sampler=None):
    """Serve only while collecting; stop and join before any optimizer update."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = {'episodes': {}, 'errors': []}
    # Fast tokenizers may raise when one instance is used from several threads.
    tokenizer_lock = threading.Lock()
    state_lock = threading.Lock()
    chat_template = mixed_channel_template(runtime.tokenizer.chat_template)

    def append(name, row):
        with state_lock:
            with (output/name).open('a') as stream:
                stream.write(json.dumps(row)+'\n')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            encoded = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == '/v1/models' or (ROUTE.fullmatch(self.path) and self.path.endswith('/models')):
                return self.reply(200, {'object': 'list', 'data': [{
                    'id': 'on-policy-role-model', 'object': 'model', 'created': 0,
                    'owned_by': 'local', 'max_model_len': generation['context_limit']}]})
            self.reply(404, {'error': 'Unknown route'})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/episode':
                    with state_lock:
                        if body['id'] in state['episodes']:
                            raise ValueError(f"Episode {body['id']} is already registered")
                        state['episodes'][body['id']] = dict(seed=int(body['seed']), request=0, raw={},
                                                             lock=threading.Lock())
                    return self.reply(200, {'ok': True})
                route = ROUTE.fullmatch(self.path)
                if route is None or route[2] != 'chat/completions' or route[1] not in state['episodes']:
                    raise ValueError('Unknown route or unregistered episode')
                episode_id, episode = route[1], state['episodes'][route[1]]
                if state['errors']:
                    return self.reply(409, {'error': {'message': 'Serving stopped after an earlier failure',
                        'type': 'on_policy_generation_error'}})
                if body.get('stream'):
                    raise ValueError('Streaming is not supported')
                if body.get('temperature', 1) != 1 or body.get('top_p', 1) != 1:
                    raise ValueError('On-policy distribution must use temperature=1, top_p=1')
                if int(body['seed']) != episode['seed']:
                    raise ValueError(f'Request seed does not match episode route {episode_id}')
                # An episode's agent loop is sequential; the lock keeps its indices exact.
                with episode['lock']:
                    return self.reply(200, self.complete(episode_id, episode, body))
            except Exception:
                with state_lock:
                    state['errors'].append(traceback.format_exc())
                    (output/'server_errors.json').write_text(json.dumps(state['errors'], indent=2))
                self.reply(500, {'error': {'message': state['errors'][-1].splitlines()[-1],
                                           'type': 'on_policy_generation_error'}})

        def complete(self, episode_id, episode, body):
            with tokenizer_lock:
                prompt = runtime.tokenizer.apply_chat_template(
                    template_messages(body['messages'], episode['raw']), tools=body.get('tools'),
                    chat_template=chat_template,
                    tokenize=True, add_generation_prompt=True, reasoning_effort='high')
                stops = [runtime.tokenizer.convert_tokens_to_ids(x) for x in ('<|return|>', '<|call|>')]
            maximum = generation['context_limit']
            budget = min(generation['max_tokens'], maximum - len(prompt))
            if budget <= 0:
                append('episode_stops.jsonl', dict(episode_id=episode_id, policy_step=policy_step,
                    reason='serving_context_limit', prompt_tokens=len(prompt), context_limit=maximum,
                    generated_tokens=0))
                # This is an episode boundary, not a model generation or a
                # server error. Inspect evaluates the unchanged world state.
                return {'id': f'{episode_id}_context_limit',
                    'object': 'chat.completion', 'created': int(time.time()), 'model': body['model'],
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': len(prompt), 'completion_tokens': 0, 'total_tokens': len(prompt)}}
            index = episode['request']
            seed = int(body['seed']) + index * 1000003
            # Record the exact failing input even if generation fails before a
            # completion exists. These are diagnostics, not sampled data.
            append('inputs.jsonl', dict(episode_id=episode_id, generation_index=index,
                policy_step=policy_step, seed=seed, prompt_ids=prompt))
            started = time.perf_counter()
            completion = sampler(prompt, budget, stops, seed)
            # Persist raw samples before parsing; malformed outputs are auditable.
            append('requests.jsonl', dict(episode_id=episode_id, policy_step=policy_step,
                generation_index=index, seed=seed, prompt_ids=prompt, completion_ids=completion,
                seconds=time.perf_counter()-started))
            episode['request'] += 1
            parsed = sampler.__self__.parse_completion(completion, body.get('tools') or [])
            message = parsed['message']
            append('parser_events.jsonl', dict(episode_id=episode_id, generation_index=index,
                policy_step=policy_step, invalid=parsed['invalid'], issues=parsed['issues']))
            for position, call in enumerate(message.get('tool_calls', [])):
                call['id'] = f"{episode_id}_{episode['request']}_{position}"
            if message.get('tool_calls'):
                key = tuple(call['id'] for call in message['tool_calls'])
                with tokenizer_lock:
                    episode['raw'][key] = 'assistant' + runtime.tokenizer.decode(completion, skip_special_tokens=False)
            capped = completion[-1] not in stops
            reason = 'stop' if parsed['invalid'] else ('length' if capped else ('tool_calls' if message.get('tool_calls') else 'stop'))
            return {'id': f"{episode_id}_{episode['request']}",
                'object': 'chat.completion', 'created': int(time.time()), 'model': body['model'],
                'choices': [{'index': 0, 'message': message, 'finish_reason': reason}],
                'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(completion),
                          'total_tokens': len(prompt)+len(completion)}}

    server = Server(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        thread.join()
        # Joins any remaining request threads, so no retained write is cut short.
        server.server_close()
        runtime.hidden = None


def mixed_channel_template(template):
    """Only override mixed messages; ordinary template behavior stays identical.

    Raw text comes from this episode's generated token IDs, never client input.
    Restore the tool-name state for the subsequent tool response.
    """
    marker = "{%- if message.role == 'assistant' -%}"
    if template.count(marker) != 1:
        raise ValueError('Unexpected GPT-OSS template structure')
    replacement = """{%- if message.role == 'assistant' and message.raw_harmony is defined -%}
        {{- '<|start|>' + message.raw_harmony }}
        {%- set last_tool_call.name = message.tool_calls[-1].function.name %}
    {%- elif message.role == 'assistant' -%}"""
    return template.replace(marker, replacement)
