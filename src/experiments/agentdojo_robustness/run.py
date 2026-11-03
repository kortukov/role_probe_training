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
        return original(**kwargs)
    client.chat.completions.create = create
    llm = OpenAILLM(client, 'role-agentdojo', reasoning_effort='medium', temperature=1.0)
    config = PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None)
    pipeline = AgentPipeline.from_config(config)
    pipeline.name = 'vllm_parsed'
    a.output.mkdir(parents=True, exist_ok=True)
    with OutputLogger(str(a.output)):
        benchmark_suite_without_injections(pipeline, suite, a.output, False, user_tasks=users, benchmark_version='v1.2.2')
        attack = load_attack('important_instructions', suite, pipeline)
        benchmark_suite_with_injections(pipeline, suite, attack, a.output, False, user_tasks=users,
            injection_tasks=list(suite.injection_tasks)[:1] if a.smoke else None, benchmark_version='v1.2.2')
    (a.output / f'{a.suite}.complete.json').write_text(json.dumps({'suite': a.suite, 'users': users, 'smoke': a.smoke}))

if __name__ == '__main__':
    main()
