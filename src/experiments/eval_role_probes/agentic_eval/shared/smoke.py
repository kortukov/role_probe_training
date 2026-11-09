"""Verify GPT-OSS tool calling and reasoning through the Harmony proxy before benchmark execution.

The proxy accepts one reasoning effort, max_tokens and temperature per run; pass the run's values."""
import argparse
import asyncio
import json
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
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("base_url", help="Harmony proxy URL ending in /v1")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    args = parser.parse_args()
    model = get_model("vllm/openai/gpt-oss-20b", base_url=args.base_url, api_key="local",
                      config=GenerateConfig(temperature=args.temperature, max_tokens=args.max_tokens,
                                            reasoning_effort=args.reasoning_effort))
    messages, output = await model.generate_loop("Use lookup to check order abc123 and tell me its status.", tools=[lookup()])
    data = [m.model_dump(mode="json", exclude_none=True) for m in messages]
    assert any(m["role"] == "tool" for m in data), data
    assert "shipped" in output.completion.lower(), data
    print(json.dumps(data, indent=2))

asyncio.run(main())
