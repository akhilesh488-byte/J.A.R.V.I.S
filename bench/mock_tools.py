"""
Mock tools for benchmarking.
Covers every registry and system tool. Records (tool, args, time) and executes
nothing real, returning realistic strings loaded directly from tool_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from tools.tool_registry import TOOL_SIGNATURES, get_tool_schemas

ROOT_TOOL_RESULTS_PATH = os.path.join(ROOT, "tool_results.json")
BENCH_TOOL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_results.json")

ALL_KNOWN_TOOLS = set(TOOL_SIGNATURES.keys()) | {
    "manage_browser",
    "play_pause_media",
    "close_current_app",
    "get_window_status",
    "send_whatsapp_message",
    "delegate_to_researcher",
    "execute_full_project_pipeline",
}


def load_tool_results_templates(path: Optional[str] = None) -> Dict[str, str]:
    target_path = path or (ROOT_TOOL_RESULTS_PATH if os.path.exists(ROOT_TOOL_RESULTS_PATH) else BENCH_TOOL_RESULTS_PATH)
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


class BenchmarkMockTools:
    def __init__(self, names: Optional[List[str]] = None, results_path: Optional[str] = None):
        self.templates = load_tool_results_templates(results_path)
        tool_names = set(self.templates.keys()) | ALL_KNOWN_TOOLS
        self.names = list(names) if names is not None else sorted(tool_names)
        self.calls: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.calls.clear()

    def format_result(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        # Check if realistic string template is loaded
        template = self.templates.get(tool_name)
        if template:
            # If template has string format placeholders like {app_name}, format them
            if "{" in template and "}" in template:
                try:
                    class SafeDict(dict):
                        def __missing__(self, key):
                            return f"<{key}>"

                    return template.format_map(SafeDict(kwargs))
                except Exception:
                    return template
            return template

        return f"Executed {tool_name} successfully (mock)."

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
