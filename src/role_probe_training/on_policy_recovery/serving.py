"""Serve the colocated vLLM policy to concurrent episodes at synchronous barriers.

Each episode uses its own route, /episodes/<id>/v1, so generation indices, seeds,
tool-call IDs and retained generated tokens are per episode. vLLM batches the
generations. Inspect runs in its existing separate environment. Prompt rendering,
replay of earlier turns from their generated tokens, parsing and the token records
live in role_probe_training.harmony.server; nothing here uses the HF chat template.
"""
from contextlib import contextmanager

from role_probe_training.harmony import RenderConfig
from role_probe_training.harmony.server import HarmonyChatServer


def render_config(generation):
    return RenderConfig(generation.get('reasoning_effort', 'high'),
                        generation.get('conversation_start_date', RenderConfig.conversation_start_date))


@contextmanager
def serve(runtime, output, policy_step, generation, *, sampler=None):
    """Serve only while collecting; stop and join before any optimizer update."""
    if sampler is None:
        raise ValueError('Collection needs the colocated vLLM sampler')
    server = HarmonyChatServer(output, sampler, config=render_config(generation),
                               context_limit=generation['context_limit'], max_tokens=generation['max_tokens'],
                               model_name='on-policy-role-model', policy_step=policy_step, require_episodes=True)
    try:
        with server.serving() as url:
            yield url, server.state
    finally:
        runtime.hidden = None
