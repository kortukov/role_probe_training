"""CPU checks of the Harmony renderer: library equivalence, labels, replay, records.

    uv run python -m unittest src/experiments/harmony_rendering/checks.py
"""
import unittest
from pathlib import Path

import numpy as np
from openai_harmony import (Author, Conversation, DeveloperContent, Message, RenderConversationConfig, Role,
                            SystemContent, ReasoningEffort, ToolDescription)

from role_probe_training import harmony as h

TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get weather',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
CONFIG = h.RenderConfig(reasoning_effort='high', conversation_start_date='2025-12-31')
enc = h.encoding()


def completion(*messages):
    """Tokens the model would have generated for these assistant messages."""
    ids = list(enc.render_conversation(Conversation.from_messages(list(messages)),
                                       RenderConversationConfig(auto_drop_analysis=False)))
    header = len(h.tokens().assistant_header)
    assert ids[:header] == h.tokens().assistant_header
    ids = ids[header:]
    if ids[-1] == h.tokens().end:  # a final answer is generated with <|return|>
        ids[-1] = h.tokens().ret
    return ids


def assistant(text, channel, recipient=None):
    message = Message.from_role_and_content(Role.ASSISTANT, text).with_channel(channel)
    if recipient:
        message = message.with_recipient(recipient).with_content_type(h.TOOL_CALL_CONTENT_TYPE)
    return message


TURN_1 = completion(assistant('Need the tool.', 'analysis'), assistant('**Plan** call it.', 'commentary'),
                    assistant('{"city":"Paris"}', 'commentary', 'functions.get_weather'))
TURN_2 = completion(assistant('Got it.', 'analysis'), assistant('It is 20C.', 'final'))
TURNS = [{'role': 'system', 'content': 'Solve the task.'}, {'role': 'user', 'content': 'Weather in Paris? 😀 héllo'},
         {'role': 'assistant', 'tokens': TURN_1},
         {'role': 'tool', 'name': 'get_weather', 'content': '{"temperature": 20}'},
         {'role': 'assistant', 'tokens': TURN_2}, {'role': 'user', 'content': 'Thanks'}]


def library_conversation():
    system = SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH).with_conversation_start_date('2025-12-31')
    developer = DeveloperContent.new().with_instructions('Solve the task.').with_function_tools([
        ToolDescription.new('get_weather', 'Get weather', parameters=TOOLS[0]['function']['parameters'])])
    return Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system), Message.from_role_and_content(Role.DEVELOPER, developer),
        Message.from_role_and_content(Role.USER, 'Weather in Paris? 😀 héllo'),
        assistant('Need the tool.', 'analysis'), assistant('**Plan** call it.', 'commentary'),
        assistant('{"city":"Paris"}', 'commentary', 'functions.get_weather'),
        Message.from_author_and_content(Author.new(Role.TOOL, 'functions.get_weather'), '{"temperature": 20}')
            .with_recipient('assistant').with_channel('commentary'),
        assistant('Got it.', 'analysis'), assistant('It is 20C.', 'final'),
        Message.from_role_and_content(Role.USER, 'Thanks')])


class RendererChecks(unittest.TestCase):
    def test_matches_library_conversation_rendering(self):
        ours = h.completion_prompt(h.render_turns(TURNS, CONFIG, tools=TOOLS))
        reference = list(enc.render_conversation_for_completion(
            library_conversation(), Role.ASSISTANT, RenderConversationConfig(auto_drop_analysis=False)))
        self.assertEqual(ours, reference)

    def test_labels_and_sources(self):
        rendering = h.render_turns(TURNS, CONFIG, tools=TOOLS)
        kinds = [(m['kind'], m['label'], m['source'], m['channel']) for m in rendering.messages]
        self.assertEqual(kinds, [
            ('system', 'system', -1, None), ('developer', 'system', 0, None), ('tools', 'system', -1, None),
            ('user', 'user', 1, None), ('assistant', 'cot', 2, 'analysis'), ('assistant', 'assistant', 2, 'commentary'),
            ('assistant', 'assistant', 2, 'commentary'), ('tool', 'tool', 3, 'commentary'),
            ('assistant', 'cot', 4, 'analysis'), ('assistant', 'assistant', 4, 'final'), ('user', 'user', 5, None)])
        labels = rendering.labels
        ids = np.asarray(rendering.ids)
        specials = {t for t in ids.tolist() if enc.is_special_token(t)}
        for entry in rendering.messages:
            span = ids[entry['content_start']:entry['content_end']]
            self.assertFalse(any(enc.is_special_token(int(t)) for t in span), entry)
            if entry['kind'] != 'tools':  # the tools block is the tail of the developer message
                self.assertEqual(enc.decode(ids[entry['start']:entry['start'] + 1].tolist()), '<|start|>')
        # Every content token is labelled, every control token is not.
        for position, token in enumerate(ids.tolist()):
            if enc.is_special_token(token):
                self.assertEqual(labels[position], -1)
        self.assertEqual(enc.decode(ids[rendering.messages[4]['content_start']:rendering.messages[4]['content_end']].tolist()),
                         'Need the tool.')
        self.assertEqual(enc.decode(ids[rendering.messages[6]['content_start']:rendering.messages[6]['content_end']].tolist()),
                         '{"city":"Paris"}')
        self.assertIn(labels[rendering.messages[6]['content_start']], [h.ROLES.index('assistant')])
        self.assertTrue(specials)

    def test_prefix_property_and_text_offsets(self):
        earlier = h.completion_prompt(h.render_turns(TURNS[:2], CONFIG, tools=TOOLS))
        later = h.completion_prompt(h.render_turns(TURNS[:4], CONFIG, tools=TOOLS))
        self.assertEqual(later[:len(earlier) + len(TURN_1)], earlier + TURN_1)
        rendering = h.render_turns(TURNS, CONFIG, tools=TOOLS)
        text, offsets = rendering.text_and_offsets()
        self.assertEqual(text, enc.decode(rendering.ids))
        self.assertTrue((offsets[:, 1] >= offsets[:, 0]).all() and (offsets[1:, 0] == offsets[:-1, 1]).all())
        self.assertEqual(''.join(text[a:b] for a, b in offsets.tolist()), text)
        self.assertIn('😀 héllo', text)

    def test_replay_of_finished_and_capped_turns(self):
        # A turn cut off by the generation limit is closed with one <|end|>.
        capped = h.replay_turn(TURN_2[:-1])
        self.assertEqual(capped.ids, h.tokens().assistant_header + TURN_2[:-1] + [h.tokens().end])
        self.assertTrue(capped.messages[-1]['closed_by_replay'] and not capped.messages[-1]['complete'])
        # A limit that fell right after a message the model closed adds no second <|end|>.
        closed = h.encoding().encode('<|channel|>analysis<|message|>Think.<|end|>', allowed_special='all')
        self.assertEqual(h.replay_turn(closed).ids, h.tokens().assistant_header + closed)
        with self.assertRaises(ValueError):
            h.replay_turn([])
        # A turn the library cannot parse is replayed verbatim; only the messages before the failure are labelled.
        encode = lambda text: h.encoding().encode(text, allowed_special='all')
        broken = encode('<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final <|constrain|>commentary to=x y<|message|>Hi<|return|>')
        entries, error = h.walk_completion(broken)
        self.assertIsNotNone(error)
        replayed = h.replay_turn(broken)
        self.assertEqual(replayed.ids, h.tokens().assistant_header + broken[:-1] + [h.tokens().end])
        self.assertEqual(len(replayed.messages), len(entries))
        replayed = h.replay_turn(TURN_2)
        self.assertEqual(replayed.ids[-1], h.tokens().end)
        self.assertEqual(replayed.ids[:2], h.tokens().assistant_header)

    def test_final_message_with_content_type(self):
        # GPT-OSS answers `final <|constrain|>json` when its instructions ask for JSON: text, not a tool call.
        completion = h.encoding().encode('<|channel|>final <|constrain|>json<|message|>{"message":"hi"}<|return|>',
                                         allowed_special='all')
        entries, error = h.walk_completion(completion)
        self.assertIsNone(error)
        self.assertEqual((entries[0]['recipient'], entries[0]['content_type']), (None, '<|constrain|>json'))
        from role_probe_training.harmony.parse import completion_to_chat
        parsed = completion_to_chat(entries, [])
        self.assertFalse(parsed['invalid'])
        self.assertEqual(parsed['message']['content'], '{"message":"hi"}')

    def test_completion_parsing(self):
        entries, error = h.walk_completion(TURN_1)
        self.assertIsNone(error)
        self.assertEqual([(e['channel'], e['recipient'], e['complete']) for e in entries],
                         [('analysis', None, True), ('commentary', None, True), ('commentary', 'functions.get_weather', True)])
        self.assertEqual(entries[0]['start'], 0)
        self.assertEqual(entries[2]['content'], '{"city":"Paris"}')
        chat = h.completion_to_chat(entries, TOOLS)
        self.assertFalse(chat['invalid'], chat['issues'])
        self.assertEqual(chat['message']['reasoning_content'], 'Need the tool.')
        self.assertEqual(chat['message']['content'], '**Plan** call it.')
        self.assertEqual(chat['message']['tool_calls'][0]['function'], {'name': 'get_weather', 'arguments': '{"city":"Paris"}'})
        truncated, error = h.walk_completion(TURN_2[:-3])
        self.assertIsNone(error)
        self.assertFalse(truncated[-1]['complete'])
        self.assertTrue(h.completion_to_chat(entries[:1], TOOLS)['invalid'])
        self.assertIn('unadvertised_tool', h.completion_to_chat(entries, [])['issues'])
        garbled = enc.encode('<|channel|>functions.get_weather to=assistant <|constrain|>json<|message|>{"x":1}<|call|>',
                             allowed_special='all')
        entries, error = h.walk_completion(garbled)
        self.assertTrue(h.completion_to_chat(entries, TOOLS)['invalid'])
        self.assertIsNone(entries[0]['label'])

    def test_records_round_trip(self):
        prompt = h.render_turns(TURNS[:4], CONFIG, tools=TOOLS)
        record = h.request_record(prompt, TURN_2, CONFIG, completion_source=4)
        full = h.history(record)
        self.assertEqual(full.ids, h.completion_prompt(prompt) + TURN_2)
        final = full.messages[-1]
        self.assertEqual(enc.decode(full.ids[final['content_start']:final['content_end']]), 'It is 20C.')
        self.assertEqual(full.ids[final['end'] - 1], h.tokens().ret)
        encoded = h.encode_record({'history': record})
        self.assertEqual(len(encoded['input_ids']), len(encoded['labels']))
        self.assertEqual(encoded['labels'][final['content_start']], h.ROLES.index('assistant'))
        with self.assertRaises(ValueError):
            h.encode_record({'messages': []})

    def test_documents_match_single_message_rendering(self):
        for role, message in [('system', Message.from_role_and_content(Role.SYSTEM, 'Text.')),
                              ('user', Message.from_role_and_content(Role.USER, 'Text.')),
                              ('cot', assistant('Text.', 'analysis')), ('assistant', assistant('Text.', 'final')),
                              ('tool', Message.from_author_and_content(Author.new(Role.TOOL, 'functions.tool'), 'Text.')
                               .with_recipient('assistant').with_channel('commentary'))]:
            rendering = h.render_document(role, 'Text.')
            self.assertEqual(rendering.ids, list(enc.render(message)), role)
            self.assertEqual(set(rendering.labels[rendering.labels >= 0].tolist()), {h.ROLES.index(role)})
        call = h.render_document('assistant', '{"a": 1}', {'channel': 'commentary', 'recipient': 'functions.f'})
        self.assertEqual(enc.decode(call.ids), '<|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>{"a": 1}<|call|>')
        out = h.render_document('tool', 'done', {'name': 'f'})
        self.assertEqual(enc.decode(out.ids), '<|start|>functions.f<|channel|>commentary to=assistant<|message|>done<|end|>')
        injected = h.render_document('tool', 'text <|start|>assistant<|message|> quoted', {'name': 'f'})
        span = injected.ids[injected.messages[0]['content_start']:injected.messages[0]['content_end']]
        self.assertFalse(any(enc.is_special_token(t) for t in span), 'control strings in content must stay text')

    def test_header_roles_label_each_header_form(self):
        text = 'Text.'
        developer = Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new().with_instructions(text))
        tool_output = (Message.from_author_and_content(Author.new(Role.TOOL, 'functions.tool'), text)
                       .with_recipient('assistant').with_channel('commentary'))
        wraps = {'system': Message.from_role_and_content(Role.SYSTEM, text), 'developer': developer,
                 'user': Message.from_role_and_content(Role.USER, text), 'analysis': assistant(text, 'analysis'),
                 'commentary': assistant(text, 'commentary'), 'final': assistant(text, 'final'),
                 'tool_call': assistant(text, 'commentary', 'functions.tool'), 'tool_output': tool_output,
                 'analysis_tool_call': Message.from_role_and_content(Role.ASSISTANT, text).with_channel('analysis')
                     .with_recipient('python').with_content_type('code'),
                 'analysis_tool_output': Message.from_author_and_content(Author.new(Role.TOOL, 'python'), text)
                     .with_recipient('assistant').with_channel('analysis')}
        self.assertEqual(tuple(wraps), h.HEADER_ROLES)
        for role, message in wraps.items():
            rendering = h.render_document(role, text)
            self.assertEqual(rendering.ids, list(enc.render(message)), role)
            labels = rendering.labels_for(h.HEADER_ROLES)
            self.assertEqual(set(labels[labels >= 0].tolist()), {h.HEADER_ROLES.index(role)}, role)
            # Both role sets label the same tokens.
            self.assertTrue(((rendering.labels >= 0) == (labels >= 0)).all())
        self.assertEqual(enc.decode(h.render_document('developer', text).ids),
                         '<|start|>developer<|message|># Instructions\n\nText.<|end|>')
        self.assertEqual(enc.decode(h.render_document('analysis_tool_call', text).ids),
                         '<|start|>assistant to=python<|channel|>analysis <|constrain|>code<|message|>Text.<|call|>')
        self.assertEqual(enc.decode(h.render_document('analysis_tool_output', text).ids),
                         '<|start|>python to=assistant<|channel|>analysis<|message|>Text.<|end|>')
        call = h.render_document('tool_call', '{"a": 1}', {'name': 'f'})
        self.assertEqual(enc.decode(call.ids), '<|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>{"a": 1}<|call|>')
        self.assertEqual(h.render_document('tool_output', 'done', {'name': 'f'}).ids, h.render_document('tool', 'done', {'name': 'f'}).ids)
        with self.assertRaises(ValueError):
            h.render_document('user', text).labels_for(('user', 'assistant'))

    def test_header_roles_in_a_conversation(self):
        rendering = h.render_turns(TURNS, CONFIG, tools=TOOLS)
        five, header = rendering.labels, rendering.labels_for(h.HEADER_ROLES)
        names = [h.HEADER_ROLES[header[m['content_start']]] for m in rendering.messages]
        self.assertEqual(names, ['system', 'developer', 'developer', 'user', 'analysis', 'commentary', 'tool_call',
                                 'tool_output', 'analysis', 'final', 'user'])
        self.assertTrue(((five >= 0) == (header >= 0)).all())
        # Scored tokens: the library's system message and the tool-schema block count among the
        # header roles only, as `system` and `developer`.
        origin = np.asarray([rendering.messages[i]['source'] if i >= 0 else -2 for i in rendering.message_ids])
        np.testing.assert_array_equal(rendering.score_mask(), (five >= 0) & (origin >= 0))
        scored = rendering.score_mask(h.HEADER_ROLES)
        system, developer, tools = rendering.messages[:3]
        self.assertEqual([m['kind'] for m in (system, developer, tools)], ['system', 'developer', 'tools'])
        self.assertTrue(scored[system['content_start']:system['content_end']].all())
        self.assertFalse(rendering.score_mask()[system['content_start']:system['content_end']].any())
        self.assertTrue(scored[tools['content_start']:tools['content_end']].all())
        self.assertFalse(rendering.score_mask()[tools['content_start']:tools['content_end']].any())
        self.assertEqual(set(header[tools['content_start']:tools['content_end']].tolist()), {h.HEADER_ROLES.index('developer')})
        np.testing.assert_array_equal(scored, header >= 0)
        tools_only = h.render_turns(TURNS[1:2], CONFIG, tools=TOOLS)  # no instructions: the block is the whole message
        self.assertEqual([m['kind'] for m in tools_only.messages], ['system', 'tools', 'user'])
        np.testing.assert_array_equal(tools_only.score_mask(h.HEADER_ROLES), tools_only.labels_for(h.HEADER_ROLES) >= 0)
        self.assertEqual(int(tools_only.score_mask().sum()), tools_only.messages[2]['content_end'] - tools_only.messages[2]['content_start'])
        self.assertEqual({h.HEADER_ROLES[i] for i in header[scored]},
                         set(h.HEADER_ROLES) - {'analysis_tool_call', 'analysis_tool_output'})
        self.assertFalse(scored[header < 0].any())
        record = h.request_record(h.render_turns(TURNS[:4], CONFIG, tools=TOOLS), TURN_2, CONFIG, completion_source=4)
        encoded = h.encode_record({'history': record}, h.HEADER_ROLES)
        self.assertTrue(encoded['score_mask'][encoded['messages'][0]['content_start']])
        self.assertFalse(h.encode_record({'history': record})['score_mask'][encoded['messages'][0]['content_start']])
        final = encoded['messages'][-1]
        self.assertEqual(encoded['labels'][final['content_start']], h.HEADER_ROLES.index('final'))
        self.assertEqual(h.encode_record({'history': record})['labels'][final['content_start']], h.ROLES.index('assistant'))

    def test_header_role_rules(self):
        # Tool calls and tool outputs are split by channel: analysis is not meant for the user.
        self.assertEqual(h.header_role('assistant', 'analysis', 'python'), 'analysis_tool_call')
        self.assertEqual(h.header_role('assistant', 'analysis', 'functions.f'), 'analysis_tool_call')
        self.assertEqual(h.header_role('assistant', 'commentary', 'functions.f'), 'tool_call')
        self.assertEqual(h.header_role('assistant', 'final', 'functions.f'), 'tool_call')
        self.assertEqual(h.header_role('tool', 'analysis', 'assistant'), 'analysis_tool_output')
        self.assertEqual(h.header_role('tool', 'commentary', 'assistant'), 'tool_output')
        self.assertEqual(h.header_role('tool'), 'tool_output')
        for role in ('system', 'developer', 'user', 'assistant', 'tool', 'other', None):
            for channel in ('analysis', 'commentary', 'final', 'functions.x', None):
                for recipient in (None, 'functions.x'):
                    name = h.header_role(role, channel, recipient)
                    self.assertEqual(name is None, h.label_for(role, channel, recipient) is None, (role, channel, recipient))
        # A function call the model writes on the analysis channel is `analysis_tool_call`, in either
        # recipient position. The server answers every function call on the commentary channel.
        for text in ('<|channel|>analysis to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}<|call|>',
                     ' to=functions.get_weather<|channel|>analysis <|constrain|>json<|message|>{"city":"Paris"}<|call|>'):
            turns = TURNS[:2] + [{'role': 'assistant', 'tokens': enc.encode(text, allowed_special='all')},
                                 {'role': 'tool', 'name': 'get_weather', 'content': '{"temperature": 20}'}]
            rendering = h.render_turns(turns, CONFIG, tools=TOOLS)
            header = rendering.labels_for(h.HEADER_ROLES)
            self.assertEqual([h.HEADER_ROLES[header[m['content_start']]] for m in rendering.messages[-2:]],
                             ['analysis_tool_call', 'tool_output'])
            self.assertEqual([h.ROLES[rendering.labels[m['content_start']]] for m in rendering.messages[-2:]], ['cot', 'tool'])
        # A message the model wrote under another author's header has no five-role label; the header
        # roles label it by that header, and header-role scoring counts it.
        written = {'functions.get_weather to=assistant<|channel|>analysis': 'analysis_tool_output',
                   'functions.get_weather<|channel|>commentary to=assistant': 'tool_output', 'user': 'user'}
        for header_text, expected in written.items():
            generated = enc.encode('<|channel|>analysis<|message|>Run it.<|end|><|start|>' + header_text +
                                   '<|message|>made up<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>',
                                   allowed_special='all')
            record = h.request_record(h.render_turns(TURNS[:2], CONFIG, tools=TOOLS), generated, CONFIG, completion_source=2)
            five, header = h.encode_record({'history': record}), h.encode_record({'history': record}, h.HEADER_ROLES)
            message = header['messages'][-2]
            span = slice(message['content_start'], message['content_end'])
            self.assertIsNone(message['label'])
            self.assertEqual(set(five['labels'][span].tolist()), {-1})
            self.assertFalse(five['score_mask'][span].any())
            self.assertEqual({h.HEADER_ROLES[i] for i in header['labels'][span]}, {expected})
            self.assertTrue(header['score_mask'][span].all())
            # Everything the five roles label keeps its header role.
            self.assertTrue((header['labels'][five['labels'] >= 0] >= 0).all())
            self.assertTrue(h.completion_to_chat(*h.walk_completion(generated)[:1], TOOLS)['invalid'])
        # A generated message under a malformed header stays unlabelled in both sets.
        garbled = enc.encode('<|channel|>functions.get_weather to=assistant <|constrain|>json<|message|>{"x":1}<|call|>',
                             allowed_special='all')
        replayed = h.Rendering(list(h.tokens().assistant_header) + garbled, h.walk_completion(garbled)[0])
        self.assertTrue((replayed.labels_for(h.HEADER_ROLES) == -1).all())
        # Toucan documents name their function once; both tool wraps of the header roles use it.
        document = {'header': {'tool': {'name': 'f'}, 'assistant': {'channel': 'commentary', 'recipient': 'functions.f'}}}
        self.assertEqual(h.document_header(document, 'tool_call'), {'name': 'f'})
        self.assertEqual(h.document_header(document, 'tool_output'), {'name': 'f'})
        self.assertEqual(h.document_header(document, 'assistant'), document['header']['assistant'])
        self.assertIsNone(h.document_header(document, 'final'))
        self.assertIsNone(h.document_header({'text': 'x'}, 'tool_call'))
        self.assertIsNone(h.document_header(document, 'analysis_tool_call'))

    def test_hf_tokenizer_agrees(self):
        path = Path('.cache/models/gpt-oss-20b')
        if not path.exists():
            self.skipTest('model tokenizer not cached')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        rendering = h.render_turns(TURNS, CONFIG, tools=TOOLS)
        text = tokenizer.decode(rendering.ids, skip_special_tokens=False)
        self.assertEqual(tokenizer(text, add_special_tokens=False)['input_ids'], rendering.ids)


if __name__ == '__main__':
    unittest.main()
