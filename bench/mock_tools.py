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

BENCH_TOOL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_results.json")
ROOT_TOOL_RESULTS_PATH = os.path.join(ROOT, "tool_results.json")

KNOWN_APPS = {
    "notepad", "calculator", "calc", "chrome", "google chrome", "spotify",
    "vscode", "vs code", "visual studio code", "code", "terminal", "windows terminal",
    "cmd", "command prompt", "powershell", "settings", "windows settings",
    "explorer", "file explorer", "files", "word", "excel", "powerpoint",
    "slack", "discord", "zoom", "teams", "microsoft teams", "edge", "microsoft edge",
    "browser", "camera", "paint", "task manager"
}

VALID_BROWSER_ACTIONS = {
    "new_tab", "close_tab", "reload", "back", "forward", "search", "open_url"
}

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
    target_path = path or (BENCH_TOOL_RESULTS_PATH if os.path.exists(BENCH_TOOL_RESULTS_PATH) else ROOT_TOOL_RESULTS_PATH)
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
        # 1. Validation matching real tools
        if tool_name == "open_app":
            app_name = str(kwargs.get("app_name") or "").strip()
            if not app_name or app_name.lower() not in KNOWN_APPS:
                return "App not found on system."

        if tool_name == "manage_browser":
            action = str(kwargs.get("action") or "").strip().lower()
            if not action or action not in VALID_BROWSER_ACTIONS:
                return f"ERROR: Invalid action '{action}'. Valid actions: new_tab, close_tab, reload, back, forward, search, open_url"

        # 2. Return real result-string format from bench/tool_results.json
        template = self.templates.get(tool_name)
        if template:
            fmt_kwargs = dict(kwargs)
            if "contact" in fmt_kwargs and "recipient" not in fmt_kwargs:
                fmt_kwargs["recipient"] = fmt_kwargs["contact"]
            if tool_name == "set_alarm_at" and "minute" in fmt_kwargs:
                try:
                    fmt_kwargs["minute"] = f"{int(fmt_kwargs['minute']):02d}"
                except Exception:
                    pass

            if "{" in template and "}" in template:
                try:
                    class SafeDict(dict):
                        def __missing__(self, key):
                            return f"<{key}>"

                    return template.format_map(SafeDict(fmt_kwargs))
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
