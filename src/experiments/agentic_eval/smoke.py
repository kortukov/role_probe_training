"""Verify local GPT-OSS tool calling and reasoning before benchmark execution."""
import asyncio
import json
import sys
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import tool

@tool
def lookup():
    async def execute(order_id: str) -> str:
        """Look up an order.

        Args:
            order_id: Order identifier.
        """
        return '{"status": "shipped"}'
    return execute

async def main():
    model = get_model("vllm/openai/gpt-oss-20b", base_url=sys.argv[1], api_key="local",
                      config=GenerateConfig(temperature=0, max_tokens=1024, reasoning_effort="high"))
    messages, output = await model.generate_loop("Use lookup to check order abc123 and tell me its status.", tools=[lookup()])
    data = [m.model_dump(mode="json", exclude_none=True) for m in messages]
    assert any(m["role"] == "tool" for m in data), data
    assert "shipped" in output.completion.lower(), data
    print(json.dumps(data, indent=2))

asyncio.run(main())
