"""
Frozen baseline copy of the v1 regex classifier.
Deterministic patterns for simple commands. A match is executed straight through
the dispatch table with 0 LLM latency.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional, Tuple

OPEN_APP_PATTERN = re.compile(r"\bopen\s+(.+)", re.I)
ADD_TODO_PATTERN = re.compile(
    r"\badd\s+(?:a\s+)?(?:task|todo|to[- ]do)\s*(?:to\s+)?(?:my\s+list\s+)?(?:to\s+)?(.+)", re.I
)
ADD_TODO_ALT_PATTERN = re.compile(
    r"\badd\s+(.+?)\s+to\s+(?:my\s+)?(?:to[- ]?do\s+)?(?:list|tasks)\b", re.I
)
LIST_TODO_PATTERN = re.compile(
    r"\b(?:list|what'?s on|show)\s+(?:my\s+)?(?:to[- ]?do\s*list|tasks)\b", re.I
)
COMPLETE_TODO_PATTERN = re.compile(
    r"\b(?:mark|check off|complete|finish)\s+(.+?)\s+(?:as\s+)?(?:done|complete)\b", re.I
)
TIMER_PATTERN = re.compile(
    r"\bset\s+(?:a\s+)?timer\s+for\s+(\d+)\s*(second|sec|minute|min|hour|hr)s?\b", re.I
)

_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def classify_regex(text: str) -> Optional[Tuple[str, dict]]:
    """
    Returns (tool_name, args) if a regex pattern claims this command, else None.
    Pure function: executes nothing.
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

    m = TIMER_PATTERN.search(text)
    if m:
        seconds = int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
        return "set_timer", {"seconds": seconds}

    return None


class RegexBaselineRouter:
    """
    Frozen baseline router wrapping classify_regex.
    """
    name = "regex_baseline"
    model = "regex"

    def ask(self, text: str, dispatch_table: Dict[str, Any], history: Optional[list] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        decision = classify_regex(text)
        if decision is not None:
            tool_name, tool_args = decision
            t_tool_0 = time.perf_counter()
            err = None
            try:
                if tool_name in dispatch_table:
                    reply = str(dispatch_table[tool_name](**tool_args))
                else:
                    reply = f"Executed {tool_name}"
            except Exception as ex:
                reply = f"Error executing {tool_name}: {ex}"
                err = str(ex)

            tool_ms = (time.perf_counter() - t_tool_0) * 1000.0
            total_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "reply": reply,
                "tool_called": tool_name,
                "args": tool_args,
                "valid_json": True,
                "ttft_ms": 0.0,
                "llm_latency_ms": 0.0,
                "tool_latency_ms": round(tool_ms, 2),
                "total_latency_ms": round(total_ms, 2),
                "tokens_in": 0,
                "tokens_out": 0,
                "iterations": 1,
                "error": err,
                "model": "regex",
                "quant": "none",
            }

        # No regex match
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "reply": "Command unrecognized by regex baseline.",
            "tool_called": None,
            "args": {},
            "valid_json": False,
            "ttft_ms": 0.0,
            "llm_latency_ms": 0.0,
            "tool_latency_ms": 0.0,
            "total_latency_ms": round(total_ms, 2),
            "tokens_in": 0,
            "tokens_out": 0,
            "iterations": 0,
            "error": "unrecognized",
            "model": "regex",
            "quant": "none",
        }
