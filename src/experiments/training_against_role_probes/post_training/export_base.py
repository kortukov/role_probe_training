"""Export the unchanged base in BF16, matching trained expert precision."""
import argparse
import json
from pathlib import Path
from role_probe_training.training.model import RoleModel

p = argparse.ArgumentParser(__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
runtime = RoleModel('.cache/models/gpt-oss-20b',
    'experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz', 0, adapters=False)
runtime.close()
model = runtime.base
if hasattr(model.config, 'quantization_config'):
    del model.config.quantization_config
if hasattr(model, 'hf_quantizer'):
    del model.hf_quantizer
model.is_quantized = False
model.save_pretrained(a.output, safe_serialization=True, max_shard_size='4GB')
runtime.tokenizer.save_pretrained(a.output)
(a.output/'export.json').write_text(json.dumps({
    'base_revision': '6cee5e81ee83917806bbde320786a8fb61efebee',
    'dtype': 'bfloat16', 'adapters': False,
    'note': 'Unchanged checkpoint, MXFP4 expert weights dequantized to BF16, no updates.'
}, indent=2)+'\n')
