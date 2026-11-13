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

from role_probe_training.harmony import HEADER_ROLES, ROLES, encoding, render_document, walk_completion
from role_probe_training.probes.linear import RoleProbe, fit_probe
from role_probe_training.probes.evaluation import conversation_summary, metrics


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
    probe = fit_probe(x, y, metadata={'layer':12}, C=1.)
    assert (probe.predict(x) == np.asarray(ROLES)[y]).all()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'probe.npz'; probe.save(path)
        np.testing.assert_allclose(RoleProbe.load(path).predict_proba(x), probe.predict_proba(x))
    assert probe.metadata['solver'] == 'lbfgs' and probe.metadata['converged'] and probe.metadata['scaling'] == 'none'
    # A probe has one class per role of its role set and records the set in its metadata.
    n = len(HEADER_ROLES)
    yh = np.repeat(np.arange(n), 20)
    xh = np.eye(n)[yh]*8 + rng.normal(size=(len(yh), n))*.1
    header_probe = fit_probe(xh, yh, metadata={'layer':12}, roles=HEADER_ROLES, C=1.)
    assert header_probe.roles == HEADER_ROLES and probe.roles == ROLES and header_probe.weight.shape == (n, n)
    assert (header_probe.predict(xh) == np.asarray(HEADER_ROLES)[yh]).all()
    scored = metrics(yh, header_probe.predict_proba(xh), HEADER_ROLES)
    assert set(scored['by_role']) == set(HEADER_ROLES) and scored['uniform_accuracy'] == 1/n and np.shape(scored['confusion_counts']) == (n, n)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'probe.npz'; header_probe.save(path)
        assert RoleProbe.load(path).roles == HEADER_ROLES
    try:
        fit_probe(x, y, metadata={'layer':12}, roles=HEADER_ROLES, C=1.)
    except ValueError:
        pass
    else:
        raise AssertionError('Fitting must reject labels that do not cover the role set')
    # Unequal-length conversations must each get equal weight in the summary.
    rows=[{'labels':np.zeros(1,dtype=int),'probabilities':np.eye(5)[[0]]},
          {'labels':np.zeros(9,dtype=int),'probabilities':np.eye(5)[np.ones(9,dtype=int)]}]
    np.testing.assert_allclose(conversation_summary(rows)['system']['mean_probabilities'], [.5,.5,0,0,0])
    print('PASS: library-identical wrappers, content masks, Unicode, repeated text, control tokens, completion parser, probe fit and round-trip, header-role probe, aggregation')


if __name__ == '__main__':
    main()
