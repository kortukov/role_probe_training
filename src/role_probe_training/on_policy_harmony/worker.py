"""Parse in vLLM's environment; do not install vLLM into the training runtime."""
from role_probe_training.on_policy.vllm_worker import RefreshWorker
from .parser import parse_tokens


class HarmonyWorker(RefreshWorker):
    def parse_role_completion(self, tokens, tools):
        if not hasattr(self, '_role_tokenizer'):
            from transformers import AutoTokenizer
            self._role_tokenizer = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b', local_files_only=True)
        return parse_tokens(self._role_tokenizer, tokens, tools)
