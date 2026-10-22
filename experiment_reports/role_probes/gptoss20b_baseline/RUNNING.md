# Running the GPT-OSS-20B role-probe reproduction

From the repository root, synchronize the locked environment and initialize the
submodules. See [CLUSTER.md](../../../agent_docs/CLUSTER.md) for Slurm/storage rules.

```bash
export PATH="$PWD/.tools/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$PWD/.cache/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
uv sync --locked
git submodule update --init --recursive
```

The [configuration](../../../src/experiments/gptoss_role_probes/config.json)
pins model/dataset revisions and all experiment settings. Stage the pinned model
at `.cache/models/gpt-oss-20b`; this run copies an existing local cached snapshot
with symlinks dereferenced. For a fresh setup:

```bash
uv run hf download openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --include '*.json' '*.jinja' '*.safetensors' \
  --local-dir .cache/models/gpt-oss-20b
uv run python src/experiments/gptoss_role_probes/prepare.py
uv run python src/experiments/gptoss_role_probes/checks.py
mkdir -p experiment_reports/role_probes/gptoss20b_baseline/logs
sbatch src/experiments/gptoss_role_probes/smoke.sbatch
# After checking the smoke-test output:
sbatch src/experiments/gptoss_role_probes/train.sbatch
```

[prepare.py](../../../src/experiments/gptoss_role_probes/prepare.py) saves exact
sampled texts, IDs, split assignments and configuration under
`artifacts/inputs/`. The input files and raw conversations are ignored by Git;
their hashes and source revisions accompany the results. Use DATAPOOL storage
for large reusable datasets; this run streams the corpora and retains only its
small one-off sample. Never commit model weights or activation caches.

[run.py](../../../src/experiments/gptoss_role_probes/run.py) supports `--stage train`,
`--stage evaluate`, `--stage audit`, or the default `all` (train and evaluate). Run it inside a GPU allocation. Evaluation
reuses saved conversations if present; retain their matching config/provenance.
The audit independently recomputes the full training-objective gradient without
refitting. For example, `sbatch src/experiments/gptoss_role_probes/train.sbatch --stage audit`.
Both saved fits emitted line-search warnings; inspect the fit metadata and audit
before interpreting optimization quality.

For a different configuration, use a fresh output directory and prepare matching
inputs with `--config ... --output <new-output>/artifacts/inputs`.

- `probes/layer_12.npz` and `layer_16.npz`: weights and intercepts.
- Matching `.json` files: role order, block index, model revision and fit metadata.
- `metrics/`: held-out and conversational metrics plus execution provenance.
- `plots/`: heatmaps regenerated on CPU from saved metrics with
  `uv run python src/experiments/gptoss_role_probes/plot.py`.
- `artifacts/conversations.jsonl`: structured native model responses.
- `artifacts/token_predictions.jsonl`: per-token labels, predictions, probabilities,
  message index and character offsets for inspection.
- `artifacts/generation_failures.json`: incomplete generations, excluded and counted.

Reusable implementation: [probes](../../../src/role_probe_training/probes/README.md).
Example use inside a GPU allocation after loading the same model:

```python
from role_probe_training.probes import RoleProbe
from role_probe_training.probes.harmony import tokenize_messages
from role_probe_training.probes.model import extract_states
import numpy as np

probe = RoleProbe.load('experiment_reports/role_probes/gptoss20b_baseline/probes/layer_16.npz')
encoded = tokenize_messages(tokenizer, messages, bos=True)
positions = np.flatnonzero(encoded['labels'] >= 0)
states = extract_states(model, encoded['input_ids'], [probe.metadata['layer']], positions)
probabilities = probe.predict_proba(states[probe.metadata['layer']])
predicted_roles = probe.predict(states[probe.metadata['layer']])
```

`messages` uses `role` and string `content`, with `channel` for assistant analysis
or final messages. Tool messages can specify `name`; calls can specify `recipient`.
This renderer is GPT-OSS-specific: inspect/normalize future benchmark trajectories
rather than silently treating missing channel/tool metadata as ground truth.

Upstream revision: `ec333c40fd43fe991e1ebf66765051b6d7e35784`. The run metadata records
package versions, source hashes, input hash, GPU, Slurm ID, and Git revision/status.
