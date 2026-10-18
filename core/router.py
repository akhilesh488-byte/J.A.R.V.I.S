"""
Decides what to do with a transcribed voice command.

Directs all incoming commands directly to the LLM provider (OllamaManagerClient
or configured cloud model), with no hardcoded local tool interception.
Supports headless multi-turn tool calling and mock execution for benchmarks.
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tools import system_tools
from tools.mock_tools import MockTools
from tools.tool_registry import build_dispatch_table

MODEL_NAME_PATTERN = re.compile(
    r"\b(?:ask|activate|use|switch to)\s+(claude|chatgpt|gpt|gemini|openai|local|ollama)\b", re.I
)
SCREEN_PATTERN = re.compile(r"\b(?:screen|what.*(?:looking at|on my screen))\b", re.I)


def extract_requested_model(text: str):
    m = MODEL_NAME_PATTERN.search(text)
    return m.group(1) if m else None


@dataclass
class RouteResult:
    reply: str
    path: str                       # "llm" | "none"
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    router_ms: float = 0.0
    tool_ms: float = 0.0
    llm_ms: float = 0.0
    stats: dict = field(default_factory=dict)
    model: Optional[str] = None
    error: Optional[str] = None


class CommandRoutingResult(dict):
    """Result dictionary from route_command with attribute access support."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'CommandRoutingResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def route(text, model_manager, dispatch_table, force_llm=True, capture_screen=True,
          image_path=None, requested_model=None) -> RouteResult:
    """
    Directly routes all commands to the LLM provider (OllamaManagerClient or candidates),
    with no hardcoded local tool interception.
    capture_screen=False stops the router from grabbing the live screen
    (benchmarks pass fixed screenshots through image_path instead).
    """
    router_ms = 0.0

    if image_path is None and capture_screen and SCREEN_PATTERN.search(text):
        try:
            image_path = system_tools.take_screenshot()
        except RuntimeError as e:
            return RouteResult(reply=str(e), path="none", router_ms=router_ms, error=str(e))

    requested = requested_model or extract_requested_model(text)
    candidates = model_manager.candidates(requested)
    if not candidates:
        return RouteResult(
            reply=("I don't have an AI model set up yet. Add an API key to .env, "
                   "or set LOCAL_LLM_MODEL to use Ollama."),
            path="none", router_ms=router_ms, error="no_model")

    # Try the first candidate; fall through to the next on an API error,
    # except in benchmark mode where errors must be recorded, not hidden.
    tries = candidates[:1] if config.BENCHMARK_MODE else candidates
    last_err = None
    for client in tries:
        try:
            raw_res = client.ask(text, dispatch_table, image_path=image_path)
            stats = dict(client.last_stats)
            reply_str = raw_res.get("reply", "") if isinstance(raw_res, dict) else str(raw_res)
            llm_ms = stats.get("llm_latency_ms", stats.get("latency_ms", 0.0))
            tool_calls = stats.get("tool_calls") or []
            tool_called = stats.get("tool_called") or (tool_calls[0]["tool"] if tool_calls else None)
            tool_args = stats.get("args") or (tool_calls[0]["args"] if tool_calls else {})
            tool_ms = stats.get("tool_latency_ms", 0.0)
            return RouteResult(reply=reply_str, path="llm", tool=tool_called, args=tool_args,
                               router_ms=router_ms, tool_ms=tool_ms, llm_ms=llm_ms, stats=stats,
                               model=stats.get("model") or getattr(client, "model", None) or client.name)
        except Exception as e:
            last_err = f"{client.name}: {e}"
    return RouteResult(reply="I couldn't reach any AI model just now.", path="llm",
                       router_ms=router_ms, error=last_err,
                       model=tries[0].name if tries else None,
                       stats=dict(tries[0].last_stats) if tries else {})


def handle_command(text: str, model_manager, speak_fn) -> str:
    """Normal (non-benchmark) entry point: returns the text to speak."""
    dispatch_table = build_dispatch_table(speak_fn)
    return route(text, model_manager, dispatch_table).reply


def route_command(text: str, provider: str = "auto", use_mock: bool = False) -> CommandRoutingResult:
    """
    Headless command routing for production and benchmarks.
    Directly routes all incoming commands to OllamaManagerClient (or configured provider),
    completely bypassing the regex classifier.
    """
    t_start = time.perf_counter()
    timestamp = datetime.now().isoformat()
    prov_lower = provider.strip().lower() if provider else "auto"

    if prov_lower in ("auto", "ollama", "local", "ollama_manager"):
        from models.ollama_manager_client import OllamaManagerClient
        client = OllamaManagerClient()
        res = client.ask(text, force_mock=use_mock)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return CommandRoutingResult({
            "timestamp": timestamp,
            "text": text,
            "reply": res["reply"],
            "regex_matched": False,
            "provider": "ollama" if not res.get("escalated") else "cloud",
            "model": client.model,
            "total_ms": round(total_ms, 2),
            "llm_ms": res.get("llm_latency_ms", 0.0),
            "tool_ms": res.get("tool_latency_ms", 0.0),
            "tokens_in": res.get("tokens_in", 0),
            "tokens_out": res.get("tokens_out", 0),
            "tool_called": res.get("tool_called"),
            "args": res.get("args") or {},
            "valid_json": res.get("valid_json", False),
            "exact_match": False,
            "error": res.get("error"),
            "escalated": res.get("escalated", False),
        })

    # Cloud or alternative model via ModelManager
    from models.model_manager import ModelManager
    mgr = ModelManager()
    if use_mock:
        dispatch = MockTools().table()
    else:
        dispatch = build_dispatch_table(speak_fn=None)

    r_res = route(text, mgr, dispatch, force_llm=True, capture_screen=False, requested_model=provider)
    total_ms = (time.perf_counter() - t_start) * 1000.0
    stats = r_res.stats or {}

    return CommandRoutingResult({
        "timestamp": timestamp,
        "text": text,
        "reply": r_res.reply,
        "regex_matched": False,
        "provider": provider,
        "model": r_res.model or provider,
        "total_ms": round(total_ms, 2),
        "llm_ms": r_res.llm_ms,
        "tool_ms": r_res.tool_ms,
        "tokens_in": stats.get("prompt_tokens", 0),
        "tokens_out": stats.get("completion_tokens", 0),
        "tool_called": r_res.tool,
        "args": r_res.args or {},
        "valid_json": True if r_res.tool else False,
        "exact_match": False,
        "error": r_res.error,
        "escalated": False,
    })
