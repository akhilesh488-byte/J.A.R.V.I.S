"""
Defines the tools Jarvis exposes, both as:
  1. a dispatch table (name -> python function) for direct execution, and
  2. JSON schemas in Anthropic/standard tool-use format, so an LLM can decide
     to call them itself when a request doesn't match a simple pattern.

Adding a new capability later = write the function, add one entry here.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tools import system_tools, todo_tools, alarm_tools, gmail_tools, calendar_tools, drive_tools, memory_tools
from tools.mock_tools import MockTools

TOOL_SCHEMAS = [
    {
        "name": "open_app",
        "description": "Open an application on the user's computer by name, e.g. 'spotify', 'vscode', 'chrome'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "Name of the app to open"}
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "add_todo",
        "description": "Add a task to the user's to-do list.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "The task text"}},
            "required": ["task"],
        },
    },
    {
        "name": "list_todos",
        "description": "List the user's current (not-yet-done) to-do items.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "complete_todo",
        "description": "Mark a to-do item as done, matched by a text hint from the task.",
        "input_schema": {
            "type": "object",
            "properties": {"task_hint": {"type": "string", "description": "Words from the task to match"}},
            "required": ["task_hint"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer/reminder that fires after a number of seconds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "How many seconds until it fires"},
                "label": {"type": "string", "description": "What the reminder is about"},
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "set_alarm_at",
        "description": "Set an alarm for a specific time today or tomorrow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour, 24h format, 0-23"},
                "minute": {"type": "integer", "description": "Minute, 0-59"},
                "label": {"type": "string", "description": "What the alarm is about"},
            },
            "required": ["hour", "minute"],
        },
    },
    {
        "name": "list_recent_emails",
        "description": "List the user's most recent Gmail emails (sender and subject).",
        "input_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer", "description": "How many emails to list, default 5"}},
        },
    },
    {
        "name": "create_email_draft",
        "description": "Create a draft email in the user's Gmail for them to review and send. Does NOT send it automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "list_upcoming_events",
        "description": "List the user's upcoming Google Calendar events.",
        "input_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer", "description": "How many events to list, default 5"}},
        },
    },
    {
        "name": "create_event",
        "description": "Create a Google Calendar event. Convert any spoken relative time into ISO 8601 datetimes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title"},
                "start_time": {"type": "string", "description": "ISO 8601 start datetime, e.g. 2026-09-10T15:00:00"},
                "end_time": {"type": "string", "description": "ISO 8601 end datetime"},
                "description": {"type": "string", "description": "Optional event description"},
            },
            "required": ["summary", "start_time", "end_time"],
        },
    },
    {
        "name": "list_recent_files",
        "description": "List the user's most recently modified Google Drive files.",
        "input_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer", "description": "How many files to list, default 5"}},
        },
    },
    {
        "name": "remember_note",
        "description": "Save a short note the user asked you to remember for later conversations.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string", "description": "What to remember"}},
            "required": ["note"],
        },
    },
]

# Exact typed signatures and metadata for tool validation
TOOL_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "open_app": {"required": ["app_name"], "optional": []},
    "add_todo": {"required": ["task"], "optional": []},
    "list_todos": {"required": [], "optional": []},
    "complete_todo": {"required": ["task_hint"], "optional": []},
    "set_timer": {"required": ["seconds"], "optional": ["label"]},
    "set_alarm_at": {"required": ["hour", "minute"], "optional": ["label"]},
    "list_recent_emails": {"required": [], "optional": ["count"]},
    "create_email_draft": {"required": ["to", "subject", "body"], "optional": []},
    "list_upcoming_events": {"required": [], "optional": ["count"]},
    "create_event": {"required": ["summary", "start_time", "end_time"], "optional": ["description"]},
    "list_recent_files": {"required": [], "optional": ["count"]},
    "remember_note": {"required": ["note"], "optional": []},
}

MEMORY_TOOL_NAMES = {"remember_note"}


def validate_tool_call(tool_name: str, args: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that tool_name exists and provided args satisfy the tool signature.
    """
    if not isinstance(tool_name, str):
        return False, "Tool name must be a string."
    if tool_name not in TOOL_SIGNATURES:
        return False, f"Unknown tool: '{tool_name}'."
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return False, f"Tool arguments must be a dictionary, got {type(args).__name__}."

    sig = TOOL_SIGNATURES[tool_name]
    missing = [req for req in sig["required"] if req not in args]
    if missing:
        return False, f"Missing required arguments for {tool_name}: {missing}"

    allowed = set(sig["required"]) | set(sig["optional"])
    unexpected = [k for k in args if k not in allowed]
    if unexpected:
        return False, f"Unexpected arguments for {tool_name}: {unexpected}"

    return True, None


def get_tool_schemas():
    if config.BENCHMARK_MODE or not config.USE_MEMORY:
        return [t for t in TOOL_SCHEMAS if t["name"] not in MEMORY_TOOL_NAMES]
    return TOOL_SCHEMAS


def build_mock():
    """A MockTools covering every tool the LLM can see in benchmark mode."""
    return MockTools([t["name"] for t in get_tool_schemas()])


def build_dispatch_table(speak_fn=None, mock=None):
    """
    Returns {tool_name: callable}. Alarm tools are pre-wired with speak_fn so
    they can announce themselves. If a MockTools is passed, every tool is
    replaced by a recording stub (benchmark use).
    """
    if mock is not None:
        return mock.table()
    return {
        "remember_note": lambda note: memory_tools.remember_note(note),
        "open_app": lambda app_name: system_tools.open_app(app_name),
        "add_todo": lambda task: todo_tools.add_todo(task),
        "list_todos": lambda: todo_tools.list_todos(),
        "complete_todo": lambda task_hint: todo_tools.complete_todo(task_hint),
        "set_timer": lambda seconds, label=None: alarm_tools.set_timer(seconds, label or "", speak_fn),
        "set_alarm_at": lambda hour, minute, label=None: alarm_tools.set_alarm_at(
            hour, minute, label or "", speak_fn
        ),
        "list_recent_emails": lambda count=None: gmail_tools.list_recent_emails(count if count is not None else 5),
        "create_email_draft": lambda to, subject, body: gmail_tools.create_email_draft(to, subject, body),
        "list_upcoming_events": lambda count=None: calendar_tools.list_upcoming_events(count if count is not None else 5),
        "create_event": lambda summary, start_time, end_time, description=None: calendar_tools.create_event(
            summary, start_time, end_time, description or ""
        ),
        "list_recent_files": lambda count=None: drive_tools.list_recent_files(count if count is not None else 5),
    }
