"""CPU checks against independently computed reverse-KL values and gradients."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from role_probe_training.on_policy.data import schedule, validate_requests
from role_probe_training.on_policy.objective import reverse_kl_sum
from role_probe_training.on_policy.serving import parse_completion, template_messages


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 7).requires_grad_(False)
        self.adapter = torch.nn.Parameter(torch.randn(7)*.7)
        self.active = True

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids).cumsum(1)
        if self.active:
            hidden = hidden+self.adapter
        return SimpleNamespace(last_hidden_state=hidden)

    @contextmanager
    def disable_adapter(self):
        self.active = False
        try:
            yield
        finally:
            self.active = True


def main():
    torch.manual_seed(1)
    backbone = Backbone()
    head = torch.nn.Linear(7, 11, bias=False).requires_grad_(False)
    runtime = SimpleNamespace(model=backbone, device='cpu', hidden=None,
        base=SimpleNamespace(model=backbone, lm_head=head,
                             config=SimpleNamespace(max_position_embeddings=128)))
    request = dict(prompt_ids=[1, 2, 3], completion_ids=[4, 5])
    actual = reverse_kl_sum(runtime, request, chunk_size=1)
    actual.backward()
    gradient = backbone.adapter.grad.clone()
    backbone.adapter.grad = None
    # Position 2 predicts token 4 and position 3 predicts token 5. Neither the
    # earlier prompt positions nor the final completion state contribute.
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with backbone.disable_adapter(), torch.no_grad():
        p = F.softmax(head(backbone(ids).last_hidden_state[:, 2:4]), -1)
    log_q = F.log_softmax(head(backbone(ids).last_hidden_state[:, 2:4]), -1)
    expected = (log_q.exp()*(log_q-p.log())).sum()
    forward = (p*(p.log()-log_q)).sum()
    assert abs(expected.item()-forward.item()) > 1e-4
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gradient, backbone.adapter.grad)
    assert head.weight.grad is None and backbone.embedding.weight.grad is None
    # Numerical check independently covers the weighting derivative through q.
    original = backbone.adapter.detach().clone()
    epsilon = .001
    with torch.no_grad():
        backbone.adapter[0] += epsilon
    plus = reverse_kl_sum(runtime, request).item()
    with torch.no_grad():
        backbone.adapter[0] -= 2*epsilon
    minus = reverse_kl_sum(runtime, request).item()
    assert abs((plus-minus)/(2*epsilon)-gradient[0].item()) < 2e-4
    with torch.no_grad():
        backbone.adapter.zero_()
    torch.testing.assert_close(reverse_kl_sum(runtime, request), torch.tensor(0.), atol=1e-6, rtol=0)
    backbone.adapter.grad = None
    reverse_kl_sum(runtime, request).backward()
    assert backbone.adapter.grad.abs().max() < 1e-6
    message = parse_completion('<|channel|>analysis<|message|>Think.<|end|>'
        '<|start|>assistant to=functions.python<|channel|>commentary json<|message|>'
        '{"code":"print(1)"}<|call|>')
    assert message['reasoning_content'] == 'Think.'
    assert json.loads(message['tool_calls'][0]['function']['arguments']) == {'code': 'print(1)'}
    converted = template_messages([message])[0]
    assert converted['thinking'] == 'Think.'
    assert converted['tool_calls'][0]['function']['arguments'] == {'code': 'print(1)'}
    message['content'] = '\n'
    converted = template_messages([message])[0]
    assert converted['content'] == '' and converted['thinking'] == 'Think.'
    assert message['content'] == '\n', 'Normalization mutated the saved message'
    message['content'] = 'Meaningful commentary'
    assert template_messages([message])[0]['content'] == 'Meaningful commentary'
    assert parse_completion('<|channel|>final<|message|>Done.<|return|>')['content'] == 'Done.'
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root/'train.jsonl'
        source.write_text(''.join(json.dumps({'sample_id': str(i % 90)})+'\n' for i in range(2700)))
        dev = root/'dev.jsonl'
        dev.write_text('{"sample_id": "heldout"}\n')
        batches = schedule(source, dev)
        assert len(batches) == 169 and len(batches[-1]) == 12
        assert batches == schedule(source, dev)
        assert len({x['id'] for batch in batches for x in batch}) == 2700
        dev.write_text('{"sample_id": "0"}\n')
        try:
            schedule(source, dev)
        except ValueError:
            pass
        else:
            raise AssertionError('Task leakage was accepted')
    sample = dict(episode_id='a', policy_step=2, generation_index=0, **request)
    validate_requests([sample], [{'id': 'a'}], 2)
    try:
        validate_requests([sample], [{'id': 'a'}], 3)
    except ValueError:
        pass
    else:
        raise AssertionError('Stale policy was accepted')
    print('PASS: reverse-KL direction/value/gradient/causal shift, zero-adapter KL, '
          'frozen base, Harmony tool/reasoning parsing, complete task schedule and stale-policy rejection.')


if __name__ == '__main__':
    main()
