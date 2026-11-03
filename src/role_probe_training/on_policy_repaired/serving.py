"""Local serial HF serving: sample with the very same adapters used to train.

No export, quantization change, or asynchronous stale policy is involved. Inspect
runs in its existing separate environment and sends OpenAI-compatible requests.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import threading
import time
import traceback

import torch


def parse_completion(text):
    """Interpret generated Harmony framing as assistant channel/tool metadata."""
    # Generation starts after the template's open '<|start|>assistant' header.
    pieces = re.split(r'<\|start\|>', 'assistant' + text)
    thinking, content, calls = [], [], []
    for piece in pieces:
        if not piece:
            continue
        if '<|message|>' not in piece:
            # A generation cap can interrupt a header; preserve raw IDs in logs.
            continue
        header, body = piece.split('<|message|>', 1)
        if not header.startswith('assistant'):
            raise ValueError('Generated non-assistant Harmony message')
        body = re.split(r'<\|(?:end|return|call)\|>', body, maxsplit=1)[0]
        channel = header.split('<|channel|>', 1)[1].split()[0] if '<|channel|>' in header else 'final'
        recipient = re.search(r'\bto=([^\s<]+)', header)
        if recipient:
            name = recipient.group(1)
            if not name.startswith('functions.'):
                raise ValueError(f'Unsupported generated recipient: {name}')
            calls.append({'id': f'call_{len(calls)}', 'type': 'function',
                          'function': {'name': name.removeprefix('functions.'), 'arguments': body}})
        elif channel == 'analysis':
            thinking.append(body)
        elif channel in ('final', 'commentary'):
            content.append(body)
        else:
            raise ValueError(f'Unsupported assistant channel: {channel}')
    message = {'role': 'assistant', 'content': '\n'.join(content)}
    if thinking:
        message['reasoning_content'] = '\n'.join(thinking)
    if calls:
        message['tool_calls'] = calls
    return message


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
        if message.get('tool_calls') and message.get('thinking') and message.get('content'):
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
    state = {'episode': None, 'request': 0, 'errors': []}
    raw_completions = {}
    chat_template = mixed_channel_template(runtime.tokenizer.chat_template)

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
                    raw_completions.clear()
                    return self.reply(200, {'ok': True})
                if self.path != '/v1/chat/completions' or state['episode'] is None:
                    raise ValueError('Unknown route or no active episode')
                if state['errors']:
                    return self.reply(409, {'error': {'message': 'Serving stopped after an earlier failure',
                        'type': 'on_policy_generation_error'}})
                if body.get('stream'):
                    raise ValueError('Streaming is not supported')
                if body.get('temperature', 1) != 1 or body.get('top_p', 1) != 1:
                    raise ValueError('On-policy distribution must use temperature=1, top_p=1')
                prompt = runtime.tokenizer.apply_chat_template(
                    template_messages(body['messages'], raw_completions), tools=body.get('tools'),
                    chat_template=chat_template,
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
                if sampler is not None:
                    completion = sampler(prompt, budget, stops, seed)
                else:
                    with torch.random.fork_rng(devices=[runtime.device]), torch.no_grad():
                        torch.manual_seed(seed)
                        sampled = runtime.model.generate(
                            input_ids=ids, attention_mask=torch.ones_like(ids),
                            do_sample=True, temperature=1., top_p=1., top_k=0,
                            repetition_penalty=1., max_new_tokens=budget,
                            eos_token_id=stops, pad_token_id=runtime.pad_id, use_cache=True)
                    completion = sampled[0, len(prompt):].tolist()
                runtime.hidden = None
                row = dict(episode_id=state['episode'], policy_step=policy_step,
                           generation_index=state['request'], seed=seed,
                           prompt_ids=prompt, completion_ids=completion,
                           seconds=time.perf_counter()-started)
                # Persist raw samples before parsing; malformed outputs are auditable.
                with (output/'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                state['request'] += 1
                message = parse_completion(runtime.tokenizer.decode(completion, skip_special_tokens=False))
                for index, call in enumerate(message.get('tool_calls', [])):
                    call['id'] = f"{state['episode']}_{state['request']}_{index}"
                if message.get('tool_calls'):
                    key = tuple(call['id'] for call in message['tool_calls'])
                    raw_completions[key] = 'assistant' + runtime.tokenizer.decode(completion, skip_special_tokens=False)
                capped = completion[-1] not in stops
                reason = 'length' if capped else ('tool_calls' if message.get('tool_calls') else 'stop')
                return self.reply(200, {'id': f"{state['episode']}_{state['request']}",
                    'object': 'chat.completion', 'created': int(time.time()), 'model': body['model'],
                    'choices': [{'index': 0, 'message': message, 'finish_reason': reason}],
                    'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(completion),
                              'total_tokens': len(prompt)+len(completion)}})
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
