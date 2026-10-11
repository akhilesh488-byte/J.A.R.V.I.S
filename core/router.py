"""
Decides what to do with a transcribed voice command.

1. classify_regex(): cheap deterministic patterns for simple commands
   (open app, to-do add/list/complete, optionally timers). A match is
   executed straight through the dispatch table: no LLM, no cost.
2. Everything else goes to an LLM, which sees the same tools through
   function calling and can act as well as talk.

These patterns are a v1 baseline. Tune them on the DEV split only, then
freeze them before touching the test split (see the paper blueprint).
Known weakness, kept on purpose so it can be measured: the open-app
pattern fires on things like "what is an open question".
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tools import system_tools
from tools.tool_registry import build_dispatch_table

MODEL_NAME_PATTERN = re.compile(
    r"\b(?:ask|activate|use|switch to)\s+(claude|chatgpt|gpt|gemini|openai|local|ollama)\b", re.I
)

OPEN_APP_PATTERN = re.compile(r"\bopen\s+(.+)", re.I)
ADD_TODO_PATTERN = re.compile(
    r"\badd\s+(?:a\s+)?(?:task|todo|to[- ]do)\s*(?:to\s+)?(?:my\s+list\s+)?(?:to\s+)?(.+)", re.I)
ADD_TODO_ALT_PATTERN = re.compile(
    r"\badd\s+(.+?)\s+to\s+(?:my\s+)?(?:to[- ]?do\s+)?(?:list|tasks)\b", re.I)
LIST_TODO_PATTERN = re.compile(
    r"\b(?:list|what'?s on|show)\s+(?:my\s+)?(?:to[- ]?do\s*list|tasks)\b", re.I)
COMPLETE_TODO_PATTERN = re.compile(
    r"\b(?:mark|check off|complete|finish)\s+(.+?)\s+(?:as\s+)?(?:done|complete)\b", re.I)
TIMER_PATTERN = re.compile(
    r"\bset\s+(?:a\s+)?timer\s+for\s+(\d+)\s*(second|sec|minute|min|hour|hr)s?\b", re.I)
SCREEN_PATTERN = re.compile(r"\b(?:screen|what.*(?:looking at|on my screen))\b", re.I)

_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def extract_requested_model(text: str):
    m = MODEL_NAME_PATTERN.search(text)
    return m.group(1) if m else None


def classify_regex(text: str):
    """
    Returns (tool_name, args) if a regex pattern claims this command, else
    None. Pure function: executes nothing, so it can be used to build the
    router's confusion matrix without side effects.
    """
    m = OPEN_APP_PATTERN.search(text)
    if m and "screen" not in text.lower():
        return "open_app", {"app_name": m.group(1).strip()}

    m = ADD_TODO_PATTERN.search(text)
    if m:
        return "add_todo", {"task": m.group(1).strip()}
    m = ADD_TODO_ALT_PATTERN.search(text)
    if m:
        return "add_todo", {"task": m.group(1).strip()}

    if LIST_TODO_PATTERN.search(text):
        return "list_todos", {}

    m = COMPLETE_TODO_PATTERN.search(text)
    if m:
        return "complete_todo", {"task_hint": m.group(1).strip()}

    if config.ENABLE_TIMER_REGEX:
        m = TIMER_PATTERN.search(text)
        if m:
            seconds = int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
            return "set_timer", {"seconds": seconds}

    return None


@dataclass
class RouteResult:
    reply: str
    path: str                       # "regex" | "llm" | "none"
    tool: Optional[str] = None      # regex path: the tool that was called
    args: dict = field(default_factory=dict)
    router_ms: float = 0.0          # time to decide (regex matching)
    tool_ms: float = 0.0            # regex path: time to run the tool
    llm_ms: float = 0.0             # LLM path: time inside API calls
    stats: dict = field(default_factory=dict)  # last client stats (tokens, iterations, calls)
    model: Optional[str] = None
    error: Optional[str] = None


def route(text, model_manager, dispatch_table, force_llm=False, capture_screen=True,
          image_path=None, requested_model=None) -> RouteResult:
    """
    force_llm=True skips the regex step (the always-LLM baseline).
    capture_screen=False stops the router from grabbing the live screen
    (benchmarks pass fixed screenshots through image_path instead).
    """
    t0 = time.perf_counter()
    decision = None if force_llm else classify_regex(text)
    router_ms = (time.perf_counter() - t0) * 1000.0

    if decision:
        tool, args = decision
        t1 = time.perf_counter()
        try:
            reply = str(dispatch_table[tool](**args))
            err = None
        except Exception as e:
            reply, err = f"That didn't work: {e}", str(e)
        tool_ms = (time.perf_counter() - t1) * 1000.0
        return RouteResult(reply=reply, path="regex", tool=tool, args=args,
                           router_ms=router_ms, tool_ms=tool_ms, error=err)

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
            reply = client.ask(text, dispatch_table, image_path=image_path)
            stats = dict(client.last_stats)
            return RouteResult(reply=reply, path="llm", router_ms=router_ms,
                               llm_ms=stats.get("latency_ms", 0.0), stats=stats,
                               model=stats.get("model") or client.name)
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
