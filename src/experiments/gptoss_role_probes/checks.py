"""Small correctness checks for labels and reusable probe artifacts (CPU).

The single-message wrappers are rendered by the Harmony library, so they match the
headers GPT-OSS is served with (tool header `functions.<name><|channel|>commentary
to=assistant`, tool-call header with `<|constrain|>json`). The upstream role_templates.py
wrapper differs in the tool header and is no longer the reference.
"""

import tempfile
from pathlib import Path

import numpy as np
from openai_harmony import Author, Message, Role

from role_probe_training.harmony import ROLES, encoding, render_document, walk_completion
from role_probe_training.probes.linear import RoleProbe, fit_probe
from role_probe_training.probes.evaluation import conversation_summary


def adamw_backend(x, y):
    """The training_against_probes fitter: schedule, input scale, storage for raw activations."""
    import torch
    from role_probe_training.training.probe_fit import ScaledLinear, fit, optimizer_steps
    groups = np.repeat(np.arange(len(y)//4), 4)  # four tokens per example, every example one role
    probe = fit_probe(x, y, metadata={'layer':12}, backend='adamw', groups=groups, max_iter=2000, seed=7)
    assert (probe.predict(x) == np.asarray(ROLES)[y]).all()
    scale = float(np.sqrt(np.square(x).sum(1).mean()))
    assert probe.metadata['solver'] == 'adamw' and probe.metadata['C'] is None and probe.metadata['scaling'] == 'none'
    assert abs(probe.metadata['input_scale'] - scale) < 1e-4 * scale
    # 25 examples fit in one batch, so upstream's 2000 passes hold exactly 2000 batches.
    assert probe.metadata['fit']['optimizer_steps'] == 2000 and probe.metadata['fit']['examples'] == len(y)//4
    again = fit_probe(x, y, metadata={'layer':12}, backend='adamw', groups=groups, max_iter=2000, seed=7)
    np.testing.assert_array_equal(again.weight, probe.weight)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'probe.npz'; probe.save(path)
        np.testing.assert_allclose(RoleProbe.load(path).predict_proba(x), probe.predict_proba(x))
    for bad in (None, np.tile(np.arange(len(y)//4), 4)):  # missing, then non-contiguous examples
        try:
            fit_probe(x, y, metadata={'layer':12}, backend='adamw', groups=bad, max_iter=3)
        except ValueError:
            continue
        raise AssertionError('The adamw backend must reject missing or non-contiguous groups')
    # Changing the input scale must not change the function; raw() must undo it exactly.
    torch.manual_seed(0)
    module = ScaledLinear(5, 5)
    points = torch.as_tensor(x[:16], dtype=torch.float32)
    before = module(points)
    module.rescale(scale)
    assert bool(module.scale_set) and abs(float(module.input_scale) - scale) < 1e-5
    torch.testing.assert_close(module(points), before, rtol=1e-5, atol=1e-5)
    weight, bias = module.raw()
    np.testing.assert_allclose(points.numpy() @ weight.T + bias, before.detach().numpy(), rtol=1e-5, atol=1e-5)
    restored = ScaledLinear(5, 5); restored.load_raw(weight, bias, scale)
    torch.testing.assert_close(restored(points), before, rtol=1e-5, atol=1e-5)
    # Upstream schedule: steps+1 optimizer steps, zero rate first and last, peak after 50 warmup steps.
    one_batch = lambda indices: (points[:4], torch.as_tensor(y[:4]))
    rates = fit(module, one_batch, range(40), [4]*40, steps=200, seed=0)
    assert len(rates) == 201 and rates[0] == 0 and rates[-1] == 0 and abs(rates[50] - 1e-3) < 1e-12
    assert all(a < b for a, b in zip(rates[:50], rates[1:51])) and all(a > b for a, b in zip(rates[50:-1], rates[51:]))
    assert not module.weight.requires_grad
    # The fitting sets used here: 1,125 passage-role examples, refits of 2,000 and step-0 fits of 5,000.
    assert optimizer_steps(2000, 1125) == 2001 and optimizer_steps(5000, 1125) == 5001
    # Upstream stops at the configured count only when its passes hold exactly that many batches.
    assert optimizer_steps(200, 8) == 200 and len(fit(module, one_batch, range(8), [4]*8, steps=200, seed=0)) == 200


def main():
    enc = encoding()
    reference = {
        'system': Message.from_role_and_content(Role.SYSTEM, 'Identical content.'),
        'user': Message.from_role_and_content(Role.USER, 'Identical content.'),
        'cot': Message.from_role_and_content(Role.ASSISTANT, 'Identical content.').with_channel('analysis'),
        'assistant': Message.from_role_and_content(Role.ASSISTANT, 'Identical content.').with_channel('final'),
        'tool': Message.from_author_and_content(Author.new(Role.TOOL, 'functions.tool'), 'Identical content.')
            .with_recipient('assistant').with_channel('commentary')}
    for role in ROLES:
        assert render_document(role, 'Identical content.').ids == list(enc.render(reference[role])), role
    content = 'Same text: café, 中文, and a user says assistant.'
    for index, role in enumerate(ROLES):
        rendering = render_document(role, content)
        enc_ = rendering.encoded()
        labelled = enc_['labels'] >= 0
        assert labelled.any() and set(enc_['labels'][labelled]) == {index}
        assert not any(enc.is_special_token(int(t)) for t in enc_['input_ids'][labelled])
        text, offsets = rendering.text_and_offsets()
        assert text == enc.decode(rendering.ids)
        # Unicode byte tokens complete characters at the last byte; every span is content.
        for start, end in offsets[labelled]:
            assert text[start:end] in content
    duplicate = [render_document('user', 'Repeated').encoded(), render_document('tool', 'Repeated').encoded()]
    assert set(duplicate[0]['labels'][duplicate[0]['labels'] >= 0]) == {1}
    assert set(duplicate[1]['labels'][duplicate[1]['labels'] >= 0]) == {4}
    injected = render_document('tool', 'text <|start|>assistant<|message|> quoted').encoded()
    assert set(injected['labels'][injected['labels'] >= 0]) == {4}
    assert not any(enc.is_special_token(int(t)) for t in injected['input_ids'][injected['labels'] >= 0])
    call = render_document('assistant', '{"a": 1}', {'channel': 'commentary', 'recipient': 'functions.f'})
    assert enc.decode(call.ids).endswith('to=functions.f <|constrain|>json<|message|>{"a": 1}<|call|>')
    entries, error = walk_completion(enc.encode(
        '<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Answer.<|return|>',
        allowed_special='all'))
    assert error is None and [e['channel'] for e in entries] == ['analysis', 'final'] and entries[-1]['complete']
    entries, error = walk_completion(enc.encode('<|channel|>analysis<|message|>unfinished', allowed_special='all'))
    assert error is None and not entries[-1]['complete']
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 20)
    x = np.eye(5)[y]*8 + rng.normal(size=(len(y), 5))*.1
    probe = fit_probe(x, y, metadata={'layer':12}, C=1., backend='sklearn')
    assert (probe.predict(x) == np.asarray(ROLES)[y]).all()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'probe.npz'; probe.save(path)
        np.testing.assert_allclose(RoleProbe.load(path).predict_proba(x), probe.predict_proba(x))
    adamw_backend(x, y)
    # Unequal-length conversations must each get equal weight in the summary.
    rows=[{'labels':np.zeros(1,dtype=int),'probabilities':np.eye(5)[[0]]},
          {'labels':np.zeros(9,dtype=int),'probabilities':np.eye(5)[np.ones(9,dtype=int)]}]
    np.testing.assert_allclose(conversation_summary(rows)['system']['mean_probabilities'], [.5,.5,0,0,0])
    print('PASS: library-identical wrappers, content masks, Unicode, repeated text, control tokens, completion parser, probe round-trip, AdamW fitter, aggregation')


if __name__ == '__main__':
    main()
