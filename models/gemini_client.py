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


def _to_gemini_tools():
    # Gemini's function-declaration format is close to Anthropic's,
    # minus the outer "input_schema" -> "parameters" rename.
    return [
        {
            "function_declarations": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
                for t in TOOL_SCHEMAS
            ]
        }
    ]


class GeminiClient(ModelClient):
    name = "gemini"

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        return bool(config.GEMINI_API_KEY)

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=config.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(
                "gemini-1.5-pro",
                system_instruction=SYSTEM_PROMPT,
                tools=_to_gemini_tools(),
            )
        return self._model

    def ask(self, prompt: str, dispatch_table: dict, image_path: str = None) -> str:
        model = self._get_model()
        chat = model.start_chat()

        content = [prompt]
        if image_path and os.path.exists(image_path):
            from PIL import Image

            content.append(Image.open(image_path))

        response = chat.send_message(content)

        for _ in range(5):
            parts = response.candidates[0].content.parts
            fn_call = next((p.function_call for p in parts if p.function_call), None)
            if not fn_call:
                return "".join(p.text for p in parts if p.text).strip()

            fn = dispatch_table.get(fn_call.name)
            try:
                result = fn(**dict(fn_call.args)) if fn else f"Unknown tool: {fn_call.name}"
            except Exception as e:
                result = f"Tool error: {e}"

            response = chat.send_message(
                {
                    "function_response": {
                        "name": fn_call.name,
                        "response": {"result": str(result)},
                    }
                }
            )

        return "I'm having trouble finishing that request — could you rephrase it?"
