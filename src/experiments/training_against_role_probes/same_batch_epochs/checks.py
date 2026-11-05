"""CPU checks for checkpoint retention and rejection of misleading report inputs."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from role_probe_training.training.train_same_batch_epochs import save_checkpoint

SCRIPT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('epoch_report', SCRIPT/'report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Runtime:
    model = torch.nn.Linear(2, 5)
    probe = torch.nn.Linear(2, 5)

    def save_probe(self, path, step):
        path.write_text(str(step))


runtime = Runtime()
# Exercise real filesystem retention with tiny weights, while replacing only
# distributed barriers and the model's unrelated Hugging Face serializer.
runtime.model.save_pretrained = lambda target: (target.mkdir(), (target/'weights').write_text('adapter'))
optimizer = torch.optim.AdamW(runtime.model.parameters())
with tempfile.TemporaryDirectory() as temporary, patch('torch.distributed.get_rank',return_value=0), patch('torch.distributed.barrier'):
    root = Path(temporary)/'checkpoints'
    config = dict(expected_trajectories=2700,retained_epochs=[2,3,5])
    for step in (330,338,340,350,507,510,845):
        save_checkpoint(runtime,optimizer,root,step,config,2)
    assert sorted(p.name for p in root.glob('step*')) == ['step00510','step00845']
    for epoch, step in ((2,338),(3,507),(5,845)):
        retained = root.parent/'epochs'/f'epoch{epoch:03d}'
        state = torch.load(retained/'training.pt',weights_only=False)
        assert state['step'] == step and state['microbatch'] == 2
        assert (retained/'adapter/weights').is_file()
        assert (retained/'layer_12.npz').read_text() == str(step)

# Ensure the optimization/refit loop has not diverged from the original trainer.
original = ast.parse(Path('src/role_probe_training/training/train_same_batch.py').read_text())
continued = ast.parse(Path('src/role_probe_training/training/train_same_batch_epochs.py').read_text())
def update_body(tree):
    main = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    block = next(x for x in main.body if isinstance(x,ast.Try))
    loop = next(x for x in block.body if isinstance(x,ast.For))
    # Last conditional changes only checkpoint scheduling; loss/refit/evaluation stay identical.
    return ast.dump(ast.Module(body=loop.body[:-1],type_ignores=[]))
assert update_body(original) == update_body(continued)

root = Path('experiment_reports/training_against_role_probes/same_batch_epochs')
for benchmark, expected in report.EXPECTED.items():
    row = json.loads((root/f'evaluations/epoch001/post_training/results/{benchmark}.json').read_text())
    report.validate(row,expected)
    mutations = [lambda x: x['probe'].__setitem__('input_sha256','wrong'),
                 lambda x: x['task'].__setitem__('samples',expected-1),
                 lambda x: x['probe']['confusion_counts'][0].__setitem__(0,0),
                 lambda x: x['probe'].__setitem__('scored_trajectories',0)]
    for mutate in mutations:
        invalid = deepcopy(row)
        mutate(invalid)
        try:
            report.validate(invalid,expected)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid evaluation accepted')
print('Checkpoint retention, unchanged update loop, complete cohorts, matrix consistency, and coverage checks passed.')
