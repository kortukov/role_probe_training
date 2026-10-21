"""Metadata-labelled retrospective Harmony replay (not serving-prompt recovery)."""
import json
from role_probe_training.probes.harmony import tokenize_messages


def replay_messages(record):
    messages, origins = [], []
    def add(message, origin):
        messages.append(message)
        origins.append(origin)
    # Tool schemas are context, not a message uttered by the user/assistant.
    if record.get('tools'):
        add({'role':'system', 'content':'Available tools:\n'+json.dumps(record['tools'], ensure_ascii=False)}, -1)
    for index, m in enumerate(record['messages']):
        role = m['role']
        if role not in ('system','user','assistant','tool'):
            raise ValueError(f'Unsupported role {role}; do not silently relabel')
        if m.get('thinking'):
            if role != 'assistant':
                raise ValueError('Reasoning attached to non-assistant')
            add({'role':'assistant','channel':'analysis','content':m['thinking']},index)
        content = m.get('content') or ''
        calls = m.get('tool_calls') or []
        raw_calls = m.get('function_calls')
        if content or not (calls or raw_calls or m.get('thinking')):
            out = {'role':role,'content':content}
            if role == 'assistant':
                out.update(channel=m.get('channel','commentary' if calls or raw_calls else 'final'), ending='<|end|>' if calls or raw_calls else '<|return|>')
            if role == 'tool':
                out['name'] = m.get('name') or 'environment'
            add(out,index)
        for call in calls:
            f = call['function']
            arguments = f['arguments']
            add({'role':'assistant','channel':'commentary','recipient':'functions.'+f['name'], 'content':arguments if isinstance(arguments,str) else json.dumps(arguments,ensure_ascii=False), 'ending':'<|call|>'},index)
        if raw_calls:
            add({'role':'assistant','channel':'commentary','content':'<function_calls>\n'+raw_calls+'\n</function_calls>', 'ending':'<|call|>'},index)
    return messages, origins


def encode_record(tokenizer, record):
    messages, origins = replay_messages(record)
    encoded = tokenize_messages(tokenizer, messages, bos=True)
    encoded['messages'] = messages
    encoded['source_message_indices'] = origins
    return encoded
