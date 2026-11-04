"""Opt-in kernel selection for HF generation; no model or dependency changes."""
from contextlib import contextmanager
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@contextmanager
def regular_flex_attention():
    """Use regular FlexAttention rather than its short-query decoding kernel.

    Keep the original masks, attention sinks, model weights and KV cache. This
    process-local setting must be installed before serving threads are started.
    It is experimental until the CUDA diagnostic and episode benchmark pass.
    """
    original = ALL_ATTENTION_FUNCTIONS['flex_attention']
    def forward(*args, **kwargs):
        options = dict(kwargs.get('kernel_options') or {})
        options['FORCE_USE_FLEX_ATTENTION'] = True
        kwargs['kernel_options'] = options
        return original(*args, **kwargs)
    ALL_ATTENTION_FUNCTIONS.register('flex_attention', forward)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register('flex_attention', original)
