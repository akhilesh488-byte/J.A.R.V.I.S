"""
Gemini client. Optional and the least tested of the four: Google's older
`google-generativeai` SDK is being replaced by `google-genai`, so check
which one is current before relying on this. It is not part of the
planned benchmark (Claude vs GPT).
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.prompt import build_system_prompt
from models.base import ModelClient, new_stats
from tools.tool_registry import get_tool_schemas

LOOP_GIVE_UP = "I'm having trouble finishing that request. Could you rephrase it?"


class GeminiClient(ModelClient):
    name = "gemini"

    def __init__(self):
        super().__init__()
        self._model = None

    def is_available(self) -> bool:
        return bool(config.GEMINI_API_KEY)

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=config.GEMINI_API_KEY)
            tools = [{"function_declarations": [
                {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
                for t in get_tool_schemas()]}]
            gen_cfg = {}
            if config.LLM_TEMPERATURE is not None:
                gen_cfg["temperature"] = config.LLM_TEMPERATURE
            self._model = genai.GenerativeModel(
                config.GEMINI_MODEL, system_instruction=build_system_prompt(),
                tools=tools, generation_config=gen_cfg)
        return self._model

    def _count(self, stats, response):
        u = getattr(response, "usage_metadata", None)
        if u:
            stats["tokens_in"] += getattr(u, "prompt_token_count", 0) or 0
            stats["tokens_out"] += getattr(u, "candidates_token_count", 0) or 0

    def ask(self, prompt: str, dispatch_table: dict, image_path: str = None) -> str:
        model = self._get_model()
        stats = new_stats(config.GEMINI_MODEL)
        self.last_stats = stats
        chat = model.start_chat()

        content = [prompt]
        if image_path and os.path.exists(image_path):
            from PIL import Image

            content.append(Image.open(image_path))

        t0 = time.perf_counter()
        response = chat.send_message(content)
        stats["latency_ms"] += (time.perf_counter() - t0) * 1000.0
        stats["iterations"] += 1
        self._count(stats, response)

        for _ in range(config.MAX_TOOL_ITERATIONS):
            parts = response.candidates[0].content.parts
            fn_call = next((p.function_call for p in parts if p.function_call), None)
            if not fn_call:
                return "".join(p.text for p in parts if p.text).strip()

            args = dict(fn_call.args)
            stats["tool_calls"].append({"tool": fn_call.name, "args": args})
            fn = dispatch_table.get(fn_call.name)
            try:
                result = fn(**args) if fn else f"Unknown tool: {fn_call.name}"
            except Exception as e:
                result = f"Tool error: {e}"

            t0 = time.perf_counter()
            response = chat.send_message({"function_response": {
                "name": fn_call.name, "response": {"result": str(result)}}})
            stats["latency_ms"] += (time.perf_counter() - t0) * 1000.0
            stats["iterations"] += 1
            self._count(stats, response)

        return LOOP_GIVE_UP
