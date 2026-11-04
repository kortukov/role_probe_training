---
license: apache-2.0
base_model: openai/gpt-oss-20b
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- role-perception
- gpt-oss
---
# GPT-OSS-20B — AppWorld same-batch KL, 27,000 trajectories

This repository contains **only the complete LoRA adapter and small loading helpers**.
The public base is `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
Training: 90 AppWorld tasks × 300 trajectories, one epoch, 1,688 updates;
rank 64, alpha 128, attention and MoE expert projections. Adapter weights remain
unchanged from training (SHA256 `43969311ed93e3b15b1859906a774faa5933303e313976214cc1a9545048fa2f`).

## Transformers / PEFT

Validated preparation environment: Python 3.12, Transformers 4.57.6, PEFT 0.20.0,
PyTorch 2.9.1. A GPU with enough memory for the dequantized 20B model is required.
The config resolves the base automatically. Dequantization and FP32 adapter
computation preserve the training setup; do not merge attention into BF16.

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, Mxfp4Config

adapter = "kortukov/gpt-oss-20b-role-perception-appworld-27000"
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter,
    dtype=torch.bfloat16,
    device_map={"": 0},
    quantization_config=Mxfp4Config(dequantize=True),
    attn_implementation="flex_attention",
)
tokenizer = AutoTokenizer.from_pretrained(
    "openai/gpt-oss-20b", revision="6cee5e81ee83917806bbde320786a8fb61efebee"
)
```

The original complete adapter includes `target_parameters` for experts; loading
only the attention tensors or using a base with unrelated quantization does not
reproduce this model.

## vLLM: reproduce the evaluated model

Validation: all **603 reconstructed model and attention-adapter tensors** are
exactly equal to the evaluated export (see `validation.json`).

The supplied launcher automatically downloads the adapter and its public base,
constructs the evaluation export **locally**, and launches vLLM. It folds expert
updates into BF16 exactly as PEFT's parameter wrappers do and retains attention
updates in FP32 via the supplied worker. No full model is stored in this Hub
repository. Local preparation needs roughly 40 GiB disk for the reconstructed
export in addition to the cached public base.

```bash
hf download kortukov/gpt-oss-20b-role-perception-appworld-27000 --local-dir role_adapter
# python: Transformers 4.57.6 + PEFT 0.20.0; vllm: 0.32.0.
# Set PREPARE_PYTHON if preparation uses a separate environment.
bash role_adapter/serve.sh ./role_model_cache \
  --max-model-len 131072 --max-num-seqs 4 --gpu-memory-utilization 0.85 \
  --attention-backend TRITON_ATTN --moe-backend triton \
  --kernel-config '{"enable_jit_warmup":false}' \
  --cudagraph-capture-sizes 1 2 4 --max-cudagraph-capture-size 4
```

Send API requests with `model="role-perception"`. The launcher always applies the
adapter; callers do not need to select a separate runtime LoRA name. It supports
one GPU, matching our evaluations. Stock `vllm serve ADAPTER_REPO` is not a
replacement for this launcher: vLLM's native LoRA path has different dtype and
expert-update arithmetic. No equivalence is claimed for it.

See `export.json` for checkpoint provenance and the project report:
https://github.com/kortukov/role_probe_training/tree/main/experiment_reports/training_against_role_probes
Role-label recovery alone does not establish prompt injection robustness.
