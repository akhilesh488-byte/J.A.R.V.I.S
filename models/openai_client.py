import base64
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.base import ModelClient
from tools.tool_registry import TOOL_SCHEMAS

SYSTEM_PROMPT = (
    "You are Jarvis, a friendly voice assistant running on the user's laptop. "
    "Keep answers short and conversational since they'll be spoken aloud. "
    "Use the available tools when the user wants an action taken. "
    "Otherwise just answer directly."
)


def _to_openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOL_SCHEMAS
    ]


class OpenAIClient(ModelClient):
    name = "openai"

    def __init__(self):
        self._client = None

    def is_available(self) -> bool:
        return bool(config.OPENAI_API_KEY)

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    def ask(self, prompt: str, dispatch_table: dict, image_path: str = None) -> str:
        client = self._get_client()

        user_content = [{"type": "text", "text": prompt}]
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        for _ in range(5):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=_to_openai_tools(),
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return (msg.content or "").strip()

            messages.append(msg)
            for call in msg.tool_calls:
                fn = dispatch_table.get(call.function.name)
                try:
                    args = json.loads(call.function.arguments)
                    result = fn(**args) if fn else f"Unknown tool: {call.function.name}"
                except Exception as e:
                    result = f"Tool error: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(result),
                    }
                )

        return "I'm having trouble finishing that request — could you rephrase it?"
