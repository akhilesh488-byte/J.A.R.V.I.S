import base64
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.prompt import build_system_prompt
from models.base import ModelClient, new_stats
from tools.tool_registry import get_tool_schemas

LOOP_GIVE_UP = "I'm having trouble finishing that request. Could you rephrase it?"


class ClaudeClient(ModelClient):
    name = "claude"

    def __init__(self):
        super().__init__()
        self._client = None

    def is_available(self) -> bool:
        return bool(config.ANTHROPIC_API_KEY)

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return self._client

    def ask(self, prompt: str, dispatch_table: dict, image_path: str = None) -> str:
        client = self._get_client()
        stats = new_stats(config.CLAUDE_MODEL)
        self.last_stats = stats

        content = [{"type": "text", "text": prompt}]
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            content.insert(0, {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": img_b64}})

        messages = [{"role": "user", "content": content}]
        system = build_system_prompt()
        tools = get_tool_schemas()

        for _ in range(config.MAX_TOOL_ITERATIONS):
            kwargs = dict(model=config.CLAUDE_MODEL, max_tokens=config.LLM_MAX_TOKENS,
                          system=system, tools=tools, messages=messages)
            if config.LLM_TEMPERATURE is not None:
                kwargs["temperature"] = config.LLM_TEMPERATURE

            t0 = time.perf_counter()
            response = client.messages.create(**kwargs)
            stats["latency_ms"] += (time.perf_counter() - t0) * 1000.0
            stats["iterations"] += 1
            stats["tokens_in"] += response.usage.input_tokens
            stats["tokens_out"] += response.usage.output_tokens

            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text").strip()

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = dict(block.input)
                stats["tool_calls"].append({"tool": block.name, "args": args})
                fn = dispatch_table.get(block.name)
                try:
                    result = fn(**args) if fn else f"Unknown tool: {block.name}"
                except Exception as e:
                    result = f"Tool error: {e}"
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(result)})
            messages.append({"role": "user", "content": results})

        return LOOP_GIVE_UP
