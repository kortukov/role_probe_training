"""Generate the fixed base-model KL anchors, retaining exact Harmony token IDs.

Each prompt is one user message rendered by this repository's Harmony renderer
(src/role_probe_training/harmony/, oss-harmony) with the fixed system message every
other prompt uses. Greedy completions are sampled by token ID from a vLLM completions
endpoint, exactly as the Harmony proxy samples trajectories; generate_kl.sbatch starts
that server. Runs in the root environment; nothing here uses the Hugging Face chat template.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from role_probe_training.harmony import LIBRARY, LIBRARY_VERSION, RenderConfig, completion_prompt, encoding, render_turns, tokens
from role_probe_training.harmony.render import special
from role_probe_training.harmony.server import VllmSampler

CONTEXT = 32768
MAX_NEW_TOKENS = 1536
SEED = 42


def render_prompt(question, config):
    """Prompt token IDs that ask the base model to answer one user message."""
    return completion_prompt(render_turns([{'role': 'user', 'content': question}], config))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True, help='prompts.json written by prepare_kl.py')
    p.add_argument('--output', type=Path, required=True, help='anchors.jsonl')
    p.add_argument('--vllm-url', required=True, help='vLLM server root, for example http://127.0.0.1:18000')
    p.add_argument('--model', default='openai/gpt-oss-20b', help='served model name')
    p.add_argument('--concurrency', type=int, default=16)
    a = p.parse_args()
    config = RenderConfig()
    prompts = json.loads(a.input.read_text())
    ids = [render_prompt(question, config) for question in prompts]
    if max(map(len, ids)) + MAX_NEW_TOKENS > CONTEXT:
        raise ValueError('KL prompt exceeds generation context; do not silently truncate')
    sampler = VllmSampler(a.vllm_url, a.model, temperature=0)
    stops = tokens().stop
    ends = set(stops) | {special('<|endoftext|>')}
    started = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as pool:
        completions = list(pool.map(lambda prompt: sampler(prompt, MAX_NEW_TOKENS, stops, SEED), ids))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w') as f:
        for i, (prompt, prefix, completion) in enumerate(zip(prompts, ids, completions)):
            f.write(json.dumps(dict(id=i, prompt=prompt, prompt_ids=prefix, completion_ids=completion,
                completion=encoding().decode(completion),
                finish_reason='stop' if completion[-1] in ends else 'length')) + '\n')
    a.output.with_suffix('.manifest.json').write_text(json.dumps(dict(samples=len(completions),
        seconds=time.perf_counter() - started, model=a.model, max_new_tokens=MAX_NEW_TOKENS,
        temperature=0, seed=SEED, rendering=config.metadata(),
        serving=f'vLLM completions by token id at {a.vllm_url}; prompts rendered by {LIBRARY} {LIBRARY_VERSION}',
        prompts_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main()
