"""Serve Inspect clients through the Harmony renderer with vLLM as the token sampler.

Inspect talks to this process as to any OpenAI-compatible chat server. The proxy renders
every prompt with oss-harmony, replays earlier assistant turns from their generated
tokens, samples completion token IDs from vLLM's completions endpoint and records prompt
and completion tokens with message spans in `<output>/requests.jsonl`. Run it in the
root environment next to the vLLM server; point the client at `--port`.
"""
import argparse
import json
import signal
import threading
from pathlib import Path

from role_probe_training.harmony import RenderConfig
from role_probe_training.harmony.server import HarmonyChatServer, VllmSampler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--vllm-url', required=True, help='vLLM server root, e.g. http://127.0.0.1:18000')
    parser.add_argument('--model', required=True, help='served model name at vLLM and the name this proxy exposes')
    parser.add_argument('--output', type=Path, required=True, help='directory for requests.jsonl and server logs')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--reasoning-effort', default='high')
    parser.add_argument('--date', default=RenderConfig.conversation_start_date, help='Harmony system message date')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='sampling temperature (1, the model default, for every collection; 0 = greedy); '
                             'every client request must use the same value')
    parser.add_argument('--context-limit', type=int, default=32768)
    parser.add_argument('--timeout', type=int, default=1800, help='seconds to wait for one vLLM completion')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'{args.output} is not empty; use a fresh directory per run')
    config = RenderConfig(args.reasoning_effort, args.date)
    sampler = VllmSampler(args.vllm_url, args.model, args.timeout, temperature=args.temperature)
    server = HarmonyChatServer(args.output, sampler, config=config, context_limit=args.context_limit,
                               max_tokens=args.max_tokens, model_name=args.model, temperature=args.temperature)
    stop = threading.Event()
    for name in ('SIGTERM', 'SIGINT'):
        signal.signal(getattr(signal, name), lambda *_: stop.set())
    with server.serving(port=args.port) as url:
        (args.output / 'server.json').write_text(json.dumps({'url': url, 'vllm_url': args.vllm_url, 'model': args.model,
            'max_tokens': args.max_tokens, 'context_limit': args.context_limit, 'temperature': args.temperature,
            'rendering': config.metadata()}, indent=2) + '\n')
        print(f'listening {url}', flush=True)
        stop.wait()
    print(json.dumps({'requests': sum(1 for _ in (args.output / 'requests.jsonl').open()) if (args.output / 'requests.jsonl').exists() else 0,
                      'errors': len(server.state['errors'])}), flush=True)


if __name__ == '__main__':
    main()
