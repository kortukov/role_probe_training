"""Avoid the failing compiled single-query sliding-window mask construction."""
from contextlib import contextmanager
from transformers import masking_utils


@contextmanager
def eager_decode_masks():
    """Build single-query masks with ordinary torch operations.

    Attention itself remains FlexAttention, including its optimized decode
    kernel. Prefill and training masks remain compiled. A single-query mask is
    linear in context length; this does not enable quadratic eager attention.
    Install before starting the server and restore after joining its thread.
    """
    original = masking_utils.create_block_mask
    def create_block_mask(*args, **kwargs):
        query_length = kwargs.get('Q_LEN', args[3] if len(args)>3 else None)
        if query_length == 1:
            kwargs['_compile'] = False
        return original(*args, **kwargs)
    masking_utils.create_block_mask = create_block_mask
    try:
        yield
    finally:
        masking_utils.create_block_mask = original
