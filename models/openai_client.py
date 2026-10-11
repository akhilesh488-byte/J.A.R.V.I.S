import base64
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.prompt import build_system_prompt
from models.base import ModelClient, new_stats
from tools.tool_registry import get_tool_schemas

LOOP_GIVE_UP = "I'm having trouble finishing that request. Could you rephrase it?"


def _to_openai_tools():
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in get_tool_schemas()]


class OpenAICompatClient(ModelClient):
    """Shared loop for OpenAI and any OpenAI-compatible server (e.g. Ollama)."""

    def __init__(self):
        super().__init__()
        self._client = None

    # subclasses fill these in
    def _api_key(self): raise NotImplementedError
    def _base_url(self): return None
    def _model_id(self): raise NotImplementedError

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._api_key(), base_url=self._base_url())
        return self._client

    def ask(self, prompt: str, dispatch_table: dict, image_path: str = None) -> str:
        client = self._get_client()
        model = self._model_id()
        stats = new_stats(model)
        self.last_stats = stats

        user_content = [{"type": "text", "text": prompt}]
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            user_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        messages = [{"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": user_content}]

        for _ in range(config.MAX_TOOL_ITERATIONS):
            kwargs = dict(model=model, messages=messages, tools=_to_openai_tools())
            if config.LLM_TEMPERATURE is not None:
                kwargs["temperature"] = config.LLM_TEMPERATURE

            t0 = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            stats["latency_ms"] += (time.perf_counter() - t0) * 1000.0
            stats["iterations"] += 1
            if response.usage:
                stats["tokens_in"] += response.usage.prompt_tokens or 0
                stats["tokens_out"] += response.usage.completion_tokens or 0

            msg = response.choices[0].message
            if not msg.tool_calls:
                return (msg.content or "").strip()

            messages.append(msg)
            for call in msg.tool_calls:
                fn = dispatch_table.get(call.function.name)
                try:
                    args = json.loads(call.function.arguments or "{}")
                except Exception:
                    args = {}
                stats["tool_calls"].append({"tool": call.function.name, "args": args})
                try:
                    result = fn(**args) if fn else f"Unknown tool: {call.function.name}"
                except Exception as e:
                    result = f"Tool error: {e}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

        return LOOP_GIVE_UP


class OpenAIClient(OpenAICompatClient):
    name = "openai"

    def is_available(self) -> bool:
        return bool(config.OPENAI_API_KEY)

    def _api_key(self): return config.OPENAI_API_KEY
    def _model_id(self): return config.OPENAI_MODEL


class LocalClient(OpenAICompatClient):
    """Ollama (or any OpenAI-compatible local server). Free and offline."""
    name = "local"

    def is_available(self) -> bool:
        return bool(config.LOCAL_LLM_MODEL)

    def _api_key(self): return "ollama"  # ignored by Ollama, required by the SDK
    def _base_url(self): return config.LOCAL_LLM_URL
    def _model_id(self): return config.LOCAL_LLM_MODEL
