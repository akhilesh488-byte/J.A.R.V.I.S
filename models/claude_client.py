import base64
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.base import ModelClient
from tools.tool_registry import TOOL_SCHEMAS

SYSTEM_PROMPT = (
    "You are Jarvis, a friendly voice assistant running on the user's laptop. "
    "Keep answers short and conversational since they'll be spoken aloud. "
    "Use the available tools when the user wants an action taken "
    "(opening an app, managing their to-do list, setting an alarm). "
    "Otherwise just answer directly."
)


class ClaudeClient(ModelClient):
    name = "claude"

    def __init__(self):
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

        content = [{"type": "text", "text": prompt}]
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            content.insert(
                0,
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
            )

        messages = [{"role": "user", "content": content}]

        # Tool-use loop: keep going while Claude wants to call tools.
        for _ in range(5):  # hard cap so a bug can't loop forever
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                # Final answer: concatenate any text blocks
                return "".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = dispatch_table.get(block.name)
                try:
                    result = fn(**block.input) if fn else f"Unknown tool: {block.name}"
                except Exception as e:
                    result = f"Tool error: {e}"
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return "I'm having trouble finishing that request — could you rephrase it?"
