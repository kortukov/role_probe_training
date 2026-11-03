"""AgentDojo native-tool evaluation, with explicit generation controls."""
import argparse
import json
from pathlib import Path

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from openai import OpenAI


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    suite = get_suite('v1.2.2', a.suite)
    users = [u for u in suite.user_tasks if not (a.suite == 'workspace' and u in {'user_task_35', 'user_task_38'})]
    if a.smoke:
        users = users[:1]
    client = OpenAI(api_key='EMPTY', base_url=f'http://127.0.0.1:{a.port}/v1', timeout=1200, max_retries=2)
    # Bind generation arguments at the HTTP client boundary; AgentDojo's stock
    # OpenAILLM does not expose max_tokens and omits temperature when it is zero.
    original = client.chat.completions.create
    def create(**kwargs):
        kwargs.update(max_tokens=8192, temperature=1.0, reasoning_effort='medium', seed=123)
        completion = original(**kwargs)
        for choice in completion.choices:
            for call in choice.message.tool_calls or []:
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args, dict):
                        raise ValueError('Tool arguments must be a JSON object')
                except (json.JSONDecodeError, ValueError) as error:
                    # Use the stock executor's invalid-tool error response. Keep
                    # the raw output in both the transcript and a diagnostic log.
                    invalid = {'function': call.function.name, 'arguments': call.function.arguments,
                               'error': str(error)}
                    a.output.mkdir(parents=True, exist_ok=True)
                    with (a.output / f'{a.suite}.invalid_calls.jsonl').open('a') as log:
                        log.write(json.dumps(invalid)+'\n')
                    call.function.name = '__invalid_tool_call__'
                    call.function.arguments = json.dumps(invalid)
        return completion
    client.chat.completions.create = create
    llm = OpenAILLM(client, 'role-agentdojo', reasoning_effort='medium', temperature=1.0)
    config = PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None)
    pipeline = AgentPipeline.from_config(config)
    pipeline.name = 'vllm_parsed'
    a.output.mkdir(parents=True, exist_ok=True)
    def attempt(user, injection=None):
        try:
            if injection is None:
                benchmark_suite_without_injections(pipeline, suite, a.output, False,
                    user_tasks=[user], benchmark_version='v1.2.2')
            else:
                benchmark_suite_with_injections(pipeline, suite, attack, a.output, False,
                    user_tasks=[user], injection_tasks=[injection], benchmark_version='v1.2.2')
        except Exception as error:
            # A single bad model output or scorer failure must not discard the
            # rest of the benchmark. Retain its diagnostic and any partial trace.
            key = f"{a.suite}/{user}/{attack.name if injection else 'none'}/{injection or 'none'}.json"
            with (a.output / f'{a.suite}.failures.jsonl').open('a') as log:
                log.write(json.dumps({'key':key, 'error':f'{type(error).__name__}: {error}'})+'\n')
    with OutputLogger(str(a.output)):
        for user in users:
            attempt(user)
        attack = load_attack('important_instructions', suite, pipeline)
        injections = list(suite.injection_tasks)[:1] if a.smoke else list(suite.injection_tasks)
        for user in users:
            for injection in injections:
                attempt(user, injection)
    (a.output / f'{a.suite}.complete.json').write_text(json.dumps({'suite': a.suite, 'users': users, 'smoke': a.smoke}))

if __name__ == '__main__':
    main()
