"""Use vLLM's installed HarmonyParser without guessing missing tool recipients."""
import json
from vllm.parser.harmony import HarmonyParser
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from openai_harmony import HarmonyError, Role


class ConfiguredHarmony(HarmonyParser):
    reasoning_parser_cls = GptOssReasoningParser
    tool_parser_cls = GptOssToolParser


def parse_tokens(tokenizer, tokens, tools):
    # Instantiate per completion: no parser state leaks across episodes.
    parser = ConfiguredHarmony(tokenizer)
    issues = []
    try:
        reasoning, content, calls = parser.parse('', None, enable_auto_tools=True, model_output_token_ids=tokens)
    except HarmonyError as error:
        reasoning, content, calls = None, None, None
        issues.append('harmony_error: '+str(error))
    allowed = {tool['function']['name'] for tool in tools}
    # Keep the library's output mapping, but flag discarded/unsupported metadata.
    try:
        messages = get_encoding().parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
        for message in messages:
            if str(message.author.role) not in ('Role.ASSISTANT', 'assistant'):
                issues.append('generated_nonassistant_role')
            if message.channel not in (None, 'analysis', 'commentary', 'final'):
                issues.append('unsupported_channel: '+str(message.channel))
    except HarmonyError as error:
        issues.append('malformed_harmony: '+str(error))
    if any(call.name not in allowed for call in calls or []):
        issues.append('unadvertised_tool')
    if not calls and not content:
        issues.append('no_action_or_final_content')
    invalid = bool(issues)
    message = dict(role='assistant', content='' if invalid else (content or ''))
    if reasoning:
        message['reasoning_content'] = reasoning
    if calls and not invalid:
        message['tool_calls'] = [dict(id=f'call_{i}',type='function',function=dict(name=call.name,arguments=call.arguments)) for i,call in enumerate(calls)]
    # An empty assistant response with no tools ends Inspect's tool loop. Its
    # AppWorld finally block scores the actual world state. Raw tokens stay in
    # requests.jsonl and continue to contribute to reverse KL, without rerolls.
    return dict(message=message, invalid=invalid, issues=issues)
