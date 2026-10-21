"""
Mock tools for benchmarking.
Covers every registry and system tool. Records (tool, args, time) and executes
nothing real, returning realistic strings loaded from bench/tool_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tool_registry import TOOL_SIGNATURES, get_tool_schemas

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_RESULTS_PATH = os.path.join(BENCH_DIR, "tool_results.json")

ALL_KNOWN_TOOLS = set(TOOL_SIGNATURES.keys()) | {
    "manage_browser",
    "play_pause_media",
    "close_current_app",
    "get_window_status",
    "send_whatsapp_message",
}


def load_tool_results_templates() -> Dict[str, str]:
    if os.path.exists(TOOL_RESULTS_PATH):
        try:
            with open(TOOL_RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


class BenchmarkMockTools:
    def __init__(self, names: Optional[List[str]] = None, results_templates: Optional[Dict[str, str]] = None):
        self.names = list(names) if names is not None else sorted(ALL_KNOWN_TOOLS)
        self.templates = results_templates or load_tool_results_templates()
        self.calls: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.calls.clear()

    def format_result(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        template = self.templates.get(tool_name)
        if not template:
            return f"Executed {tool_name} successfully (mock)."
        try:
            # Format template with provided kwargs
            class SafeDict(dict):
                def __missing__(self, key):
                    return f"<{key}>"

            return template.format_map(SafeDict(kwargs))
        except Exception:
            return template

    def table(self) -> Dict[str, Callable[..., str]]:
        def make_tool(name: str):
            def tool_fn(**kwargs):
                call_record = {
                    "tool": name,
                    "args": dict(kwargs),
                    "time": time.perf_counter(),
                }
                self.calls.append(call_record)
                return self.format_result(name, kwargs)

            return tool_fn

        return {n: make_tool(n) for n in self.names}
