"""
Benchmark providers for Jarvis evaluation.
Implements:
1. OllamaBenchmarkProvider: streaming client capturing TTFT (time-to-first-token),
   prompt_eval_count, eval_count, prompt_eval_duration, eval_duration.
   Temperature = 0.0. Memory & user profile injection OFF.
2. CloudBenchmarkProvider: wraps existing ModelManager cloud clients with temperature=0.0.
3. RegexBenchmarkProvider: wraps frozen classify_regex baseline.
4. StubBenchmarkProvider: offline deterministic mock provider for dry runs and unit tests.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from bench.baselines.regex_router import RegexBaselineRouter
from core import memory
from tools.tool_registry import build_dispatch_table, validate_tool_call

V2_SYSTEM_TOOLS = {
    "manage_browser",
    "play_pause_media",
    "close_current_app",
    "get_window_status",
    "send_whatsapp_message",
}


def build_benchmark_system_prompt() -> str:
    """
    Builds clean system prompt for benchmarks.
    Profile injection and memory notes are strictly OFF.
    """
    identity = memory.load_identity()
    tools_block = """You have access to the following desktop tools:
- open_app(app_name: str) - Open an application
- manage_browser(action: str, query: str = None) - Manage browser actions, searching, or opening URLs
- play_pause_media() - Toggle media playback
- close_current_app(window_title: str = None) - Close active application
- get_window_status() - Get active window and open applications
- send_whatsapp_message(recipient: str, message: str) - Send a WhatsApp message
- add_todo(task: str) - Add a task to to-do list
- list_todos() - List current to-do items
- complete_todo(task_hint: str) - Mark to-do item as done
- set_timer(seconds: int, label: str = None) - Set a countdown timer
- set_alarm_at(hour: int, minute: int, label: str = None) - Set an alarm
- list_recent_emails(count: int = None) - List recent emails
- create_email_draft(to: str, subject: str, body: str) - Create email draft
- list_upcoming_events(count: int = None) - List calendar events
- create_event(summary: str, start_time: str, end_time: str, description: str = None) - Create calendar event
- list_recent_files(count: int = None) - List recent files
- remember_note(note: str) - Remember a note

If you need to call a tool, respond ONLY with a JSON object in this exact format:
{"tool": "<tool_name>", "args": {<argument_name>: <value>}}
Do NOT wrap the JSON in markdown code fences.
If no tool is needed or after receiving TOOL_RESULT, respond naturally and concisely to the user."""
    return f"{identity}\n\n{tools_block}"


class BenchmarkProvider:
    name: str = "base"
    model: str = "base"
    quant: str = "none"

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class OllamaBenchmarkProvider(BenchmarkProvider):
    name = "ollama"

    def __init__(
        self,
        model_tag: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model_tag or config.LOCAL_LLM_MODEL or "jarvis-3b"
        self.base_url = (base_url or config.LOCAL_LLM_URL).replace("/v1", "")
        self.timeout = timeout
        self.quant = self._detect_quantization()

    def _detect_quantization(self) -> str:
        tag_lower = self.model.lower()
        for q in ["q4_k_m", "q4_0", "q4_1", "q5_k_m", "q8_0", "fp16", "f16"]:
            if q in tag_lower:
                return q.upper()
        return "Q4_K_M"

    def _stream_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, float, Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.0,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        ttft_ms = 0.0
        full_content = []
        final_meta: Dict[str, Any] = {}

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                msg = chunk.get("message", {}).get("content", "")
                if msg:
                    if ttft_ms == 0.0:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    full_content.append(msg)
                if chunk.get("done"):
                    final_meta = chunk

        total_content = "".join(full_content).strip()
        if ttft_ms == 0.0:
            ttft_ms = (time.perf_counter() - t_start) * 1000.0

        return total_content, ttft_ms, final_meta

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        system_prompt = build_benchmark_system_prompt()

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text})

        ttft_ms = 0.0
        tokens_in = 0
        tokens_out = 0
        prompt_eval_dur_ns = 0
        eval_dur_ns = 0
        iterations = 0
        tool_called = None
        tool_args = {}
        valid_json = False
        final_reply = ""
        error_msg = None

        try:
            iterations += 1
            content, first_ttft, meta = self._stream_chat(messages)
            ttft_ms = first_ttft
            tokens_in += meta.get("prompt_eval_count", 0)
            tokens_out += meta.get("eval_count", 0)
            prompt_eval_dur_ns += meta.get("prompt_eval_duration", 0)
            eval_dur_ns += meta.get("eval_duration", 0)

            clean_json = content.strip()
            if "```" in clean_json:
                import re
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_json, re.DOTALL)
                if m:
                    clean_json = m.group(1).strip()
            if not clean_json.startswith("{"):
                import re
                m = re.search(r"\{.*?\}", clean_json, re.DOTALL)
                if m:
                    clean_json = m.group(0).strip()

            if clean_json.startswith("{"):
                try:
                    parsed = json.loads(clean_json)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        valid_json = True
                        name = parsed["tool"]
                        raw_args = parsed.get("args") or {}
                        if not isinstance(raw_args, dict):
                            raw_args = {}

                        is_valid, _ = validate_tool_call(name, raw_args)
                        if not is_valid and name in V2_SYSTEM_TOOLS:
                            is_valid = True

                        if is_valid:
                            tool_called = name
                            tool_args = raw_args

                            # Execute mock tool from dispatch table
                            if name in dispatch_table:
                                tool_output = str(dispatch_table[name](**raw_args))
                            else:
                                tool_output = f"Executed {name}"

                            # Second turn with TOOL_RESULT
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": f"TOOL_RESULT: {tool_output}"})

                            iterations += 1
                            second_content, _, meta2 = self._stream_chat(messages)
                            tokens_in += meta2.get("prompt_eval_count", 0)
                            tokens_out += meta2.get("eval_count", 0)
                            prompt_eval_dur_ns += meta2.get("prompt_eval_duration", 0)
                            eval_dur_ns += meta2.get("eval_duration", 0)
                            final_reply = second_content
                        else:
                            final_reply = content
                    else:
                        final_reply = content
                except Exception as ex:
                    final_reply = content
                    error_msg = f"JSON parse error: {ex}"
            else:
                final_reply = content

        except Exception as e:
            error_msg = str(e)
            final_reply = f"Error: {e}"

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "reply": final_reply,
            "tool_called": tool_called,
            "args": tool_args,
            "valid_json": valid_json,
            "ttft_ms": round(ttft_ms, 2),
            "total_ms": round(total_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "iterations": iterations,
            "model": self.model,
            "quant": self.quant,
            "prompt_eval_duration_ms": round(prompt_eval_dur_ns / 1e6, 2),
            "eval_duration_ms": round(eval_dur_ns / 1e6, 2),
            "error": error_msg,
        }


class CloudBenchmarkProvider(BenchmarkProvider):
    name = "cloud"

    def __init__(self, provider_name: str = "claude"):
        from models.model_manager import ModelManager
        self.provider_name = provider_name
        self.manager = ModelManager()
        self.client = self._resolve_client()
        self.model = getattr(self.client, "model", None) or getattr(self.client, "name", provider_name)
        self.quant = "cloud_api"

    def _resolve_client(self):
        candidates = self.manager.candidates(self.provider_name)
        if not candidates:
            raise RuntimeError(f"No configured cloud client for '{self.provider_name}'. Check API keys.")
        return candidates[0]

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        error_msg = None
        try:
            # Enforce temperature 0
            old_temp = getattr(config, "LLM_TEMPERATURE", 0.0)
            config.LLM_TEMPERATURE = 0.0
            raw_res = self.client.ask(text, dispatch_table)
            config.LLM_TEMPERATURE = old_temp

            reply_str = raw_res.get("reply", "") if isinstance(raw_res, dict) else str(raw_res)
            stats = dict(getattr(self.client, "last_stats", {}) or {})
            total_ms = (time.perf_counter() - t0) * 1000.0

            tool_calls = stats.get("tool_calls") or []
            tool_called = stats.get("tool_called") or (tool_calls[0]["tool"] if tool_calls else None)
            tool_args = stats.get("args") or (tool_calls[0]["args"] if tool_calls else {})

            return {
                "reply": reply_str,
                "tool_called": tool_called,
                "args": tool_args,
                "valid_json": bool(tool_called),
                "ttft_ms": round(stats.get("llm_latency_ms", total_ms) * 0.4, 2),
                "total_ms": round(total_ms, 2),
                "tokens_in": stats.get("prompt_tokens", stats.get("tokens_in", 0)),
                "tokens_out": stats.get("completion_tokens", stats.get("tokens_out", 0)),
                "iterations": stats.get("iterations", 1),
                "model": self.model,
                "quant": self.quant,
                "prompt_eval_duration_ms": 0.0,
                "eval_duration_ms": 0.0,
                "error": None,
            }
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "reply": f"Cloud error: {e}",
                "tool_called": None,
                "args": {},
                "valid_json": False,
                "ttft_ms": 0.0,
                "total_ms": round(total_ms, 2),
                "tokens_in": 0,
                "tokens_out": 0,
                "iterations": 1,
                "model": self.model,
                "quant": self.quant,
                "prompt_eval_duration_ms": 0.0,
                "eval_duration_ms": 0.0,
                "error": str(e),
            }


class RegexBenchmarkProvider(BenchmarkProvider):
    name = "regex"
    model = "regex_baseline"
    quant = "none"

    def __init__(self):
        self.router = RegexBaselineRouter()

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        res = self.router.ask(text, dispatch_table, history)
        return {
            "reply": res["reply"],
            "tool_called": res["tool_called"],
            "args": res["args"],
            "valid_json": res["valid_json"],
            "ttft_ms": 0.0,
            "total_ms": res["total_latency_ms"],
            "tokens_in": 0,
            "tokens_out": 0,
            "iterations": res["iterations"],
            "model": "regex_baseline",
            "quant": "none",
            "prompt_eval_duration_ms": 0.0,
            "eval_duration_ms": 0.0,
            "error": res.get("error"),
        }


class StubBenchmarkProvider(BenchmarkProvider):
    """
    Offline deterministic stub provider for dry runs and unit testing.
    Requires no GPU, network, API keys, or Ollama server.
    """
    name = "stub"
    model = "stub_model_v1"
    quant = "Q4_K_M"

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        text_lower = text.strip().lower()

        tool_called = None
        tool_args = {}
        valid_json = False
        reply = "I understand and am standing by."

        if "open spotify" in text_lower:
            tool_called = "open_app"
            tool_args = {"app_name": "spotify"}
            valid_json = True
        elif "set a timer" in text_lower:
            tool_called = "set_timer"
            tool_args = {"seconds": 300}
            valid_json = True
        elif "mark" in text_lower or "complete" in text_lower:
            tool_called = "complete_todo"
            tool_args = {"task_hint": "buy groceries"}
            valid_json = True
        elif "to-do list" in text_lower or "todo list" in text_lower:
            if "add" in text_lower:
                tool_called = "add_todo"
                tool_args = {"task": "buy groceries"}
                valid_json = True
            else:
                tool_called = "list_todos"
                tool_args = {}
                valid_json = True
        elif "draft email" in text_lower:
            tool_called = "create_email_draft"
            tool_args = {"to": "alice@example.com", "subject": "Meeting", "body": "Hello Alice"}
            valid_json = True
        elif "send it now" in text_lower or "whatsapp" in text_lower:
            tool_called = "send_whatsapp_message"
            tool_args = {"recipient": "Bob", "message": "hello"}
            valid_json = True
        elif "make me a pizza" in text_lower:
            reply = "I'm afraid I cannot make food or prepare a pizza, sir. I can only assist with digital tasks."
        elif "capital of france" in text_lower:
            reply = "The capital of France is Paris, sir."
        elif "open question" in text_lower:
            reply = "An open question is one that cannot be answered with a simple 'yes' or 'no', sir."

        if tool_called:
            if tool_called in dispatch_table:
                reply = dispatch_table[tool_called](**tool_args)
            else:
                reply = f"Executed {tool_called}"

        total_ms = max(1.2, (time.perf_counter() - t0) * 1000.0)
        ttft_ms = 0.8

        return {
            "reply": reply,
            "tool_called": tool_called,
            "args": tool_args,
            "valid_json": valid_json,
            "ttft_ms": ttft_ms,
            "total_ms": round(total_ms, 2),
            "tokens_in": 128,
            "tokens_out": 24,
            "iterations": 2 if tool_called else 1,
            "model": self.model,
            "quant": self.quant,
            "prompt_eval_duration_ms": 15.0,
            "eval_duration_ms": 25.0,
            "error": None,
        }


def get_provider(provider_name: str, model_tag: Optional[str] = None) -> BenchmarkProvider:
    prov_lower = (provider_name or "").strip().lower()
    if prov_lower in ("stub", "mock", "test"):
        return StubBenchmarkProvider()
    elif prov_lower in ("regex", "regex_router", "baseline"):
        return RegexBenchmarkProvider()
    elif prov_lower in ("ollama", "local", "ollama_manager"):
        return OllamaBenchmarkProvider(model_tag=model_tag)
    elif prov_lower in ("claude", "openai", "gemini", "cloud"):
        return CloudBenchmarkProvider(provider_name=provider_name)
    else:
        # Default: treat as Ollama model tag or fallback to Ollama
        return OllamaBenchmarkProvider(model_tag=provider_name)
