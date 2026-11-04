#!/usr/bin/env bash
# PREPARE_PYTHON can point at a separate Transformers+PEFT environment.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
output=${1:?Usage: bash serve.sh LOCAL_MODEL_CACHE [vllm arguments...]}
shift
"${PREPARE_PYTHON:-python}" "$here/prepare_vllm.py" --output "$output"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec vllm serve "$output" --worker-cls role_worker.RoleWorker \
 --served-model-name role-perception --tool-call-parser openai --enable-auto-tool-choice "$@"
