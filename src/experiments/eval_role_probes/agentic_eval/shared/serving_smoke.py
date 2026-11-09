"""GPU smoke of the serving stack without Inspect; run by shared/hx2/serving_smoke.sbatch.

First samples a completion by token ID straight from vLLM, the way generate_kl.py and the
Harmony proxy do, then runs one tool round trip through the Harmony proxy, which replays
the model's own tool-call tokens in the second prompt. Runs in the root environment."""
import argparse
import json
import urllib.error
import urllib.request

from role_probe_training.harmony import RenderConfig, completion_prompt, encoding, render_turns, tokens
from role_probe_training.harmony.server import VllmSampler

TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]


def chat(url, body):
    request = urllib.request.Request(url + '/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)['choices'][0]['message']
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--vllm-url', required=True, help='vLLM server root, for example http://127.0.0.1:18000')
    p.add_argument('--proxy-url', required=True, help='Harmony proxy URL ending in /v1')
    p.add_argument('--model', default='openai/gpt-oss-20b')
    p.add_argument('--max-tokens', type=int, default=16384, help="must equal the proxy's --max-tokens")
    a = p.parse_args()
    stops = tokens().stop
    prompt = completion_prompt(render_turns([{'role': 'user', 'content': 'Name the capital of France in one word.'}], RenderConfig()))
    sampler = VllmSampler(a.vllm_url, a.model, temperature=0)
    first, second = sampler(prompt, 512, stops, 42), sampler(prompt, 512, stops, 43)
    assert first[-1] in stops, 'completion did not end at a stop token'
    print('PASS completions by token ID; repeated greedy call identical:', first == second, '| tail:',
          json.dumps(encoding().decode(first)[-100:]), flush=True)
    messages = [{'role': 'system', 'content': 'Use the tool to answer weather questions.'},
                {'role': 'user', 'content': 'What is the weather in Paris right now?'}]
    body = dict(model=a.model, messages=messages, tools=TOOLS, temperature=0, max_tokens=a.max_tokens, reasoning_effort='high')
    reply = chat(a.proxy_url, body)
    calls = reply.get('tool_calls') or []
    assert calls and calls[0]['function']['name'] == 'get_weather', reply
    json.loads(calls[0]['function']['arguments'])
    messages += [reply, {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '{"temperature_c": 18, "sky": "clear"}'}]
    final = chat(a.proxy_url, dict(body, messages=messages))
    assert not final.get('tool_calls') and '18' in (final.get('content') or ''), final
    print('PASS proxy tool round trip:', json.dumps(calls[0]['function']), '->', json.dumps(final['content'][:160]), flush=True)


if __name__ == '__main__':
    main()
