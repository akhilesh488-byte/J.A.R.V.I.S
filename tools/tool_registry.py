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

BROWSER_ACTIONS = ["new_tab", "close_tab", "reload", "back", "forward", "search", "open_url"]

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
        "name": "manage_browser",
        "description": "Manage browser actions, searching, or opening URLs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": BROWSER_ACTIONS,
                    "description": "Browser action to execute",
                },
                "query": {"type": "string", "description": "Optional search query or URL to open"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "play_pause_media",
        "description": "Toggle media playback.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "close_current_app",
        "description": "Close active application.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_window_status",
        "description": "Get active window and open applications.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "send_whatsapp_message",
        "description": "Send a WhatsApp message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "Name of the recipient or contact"},
                "recipient": {"type": "string", "description": "Name of the recipient or contact"},
                "message": {"type": "string", "description": "The message body text"},
            },
            "required": ["message"],
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
    "open_app": {
        "required": ["app_name"],
        "optional": [],
        "types": {"app_name": str},
    },
    "manage_browser": {
        "required": ["action"],
        "optional": ["query"],
        "types": {"action": str, "query": str},
        "enums": {"action": BROWSER_ACTIONS},
    },
    "play_pause_media": {
        "required": [],
        "optional": [],
        "types": {},
    },
    "close_current_app": {
        "required": [],
        "optional": [],
        "types": {},
    },
    "get_window_status": {
        "required": [],
        "optional": [],
        "types": {},
    },
    "send_whatsapp_message": {
        "required": ["message"],
        "optional": ["contact", "recipient"],
        "types": {"message": str, "contact": str, "recipient": str},
    },
    "add_todo": {
        "required": ["task"],
        "optional": [],
        "types": {"task": str},
    },
    "list_todos": {
        "required": [],
        "optional": [],
        "types": {},
    },
    "complete_todo": {
        "required": ["task_hint"],
        "optional": [],
        "types": {"task_hint": str},
    },
    "set_timer": {
        "required": ["seconds"],
        "optional": ["label"],
        "types": {"seconds": int, "label": str},
    },
    "set_alarm_at": {
        "required": ["hour", "minute"],
        "optional": ["label"],
        "types": {"hour": int, "minute": int, "label": str},
    },
    "list_recent_emails": {
        "required": [],
        "optional": ["count"],
        "types": {"count": int},
    },
    "create_email_draft": {
        "required": ["to", "subject", "body"],
        "optional": [],
        "types": {"to": str, "subject": str, "body": str},
    },
    "list_upcoming_events": {
        "required": [],
        "optional": ["count"],
        "types": {"count": int},
    },
    "create_event": {
        "required": ["summary", "start_time", "end_time"],
        "optional": ["description"],
        "types": {"summary": str, "start_time": str, "end_time": str, "description": str},
    },
    "list_recent_files": {
        "required": [],
        "optional": ["count"],
        "types": {"count": int},
    },
    "remember_note": {
        "required": ["note"],
        "optional": [],
        "types": {"note": str},
    },
}

MEMORY_TOOL_NAMES = {"remember_note"}


def validate_tool_call(tool_name: str, args: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    """
    Strict dispatcher validator: validates tool name, argument names, argument types,
    and enum values. On any violation returns (False, 'ERROR: <reason>').
    """
    if not isinstance(tool_name, str):
        return False, "ERROR: Tool name must be a string."
    if tool_name not in TOOL_SIGNATURES:
        return False, f"ERROR: Unknown tool: '{tool_name}'."
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return False, f"ERROR: Tool arguments must be a dictionary, got {type(args).__name__}."

    sig = TOOL_SIGNATURES[tool_name]

    # Required arguments check
    missing = [req for req in sig["required"] if req not in args]
    if missing:
        return False, f"ERROR: Missing required arguments for {tool_name}: {missing}"

    # Special recipient/contact check for send_whatsapp_message
    if tool_name == "send_whatsapp_message":
        if "contact" not in args and "recipient" not in args:
            return False, "ERROR: Missing required recipient or contact for send_whatsapp_message."

    # Unexpected arguments check
    allowed = set(sig["required"]) | set(sig["optional"])
    unexpected = [k for k in args if k not in allowed]
    if unexpected:
        return False, f"ERROR: Unexpected arguments for {tool_name}: {unexpected}"

    # Types check
    types = sig.get("types", {})
    for k, val in args.items():
        if val is None:
            continue
        expected_type = types.get(k)
        if expected_type == str and not isinstance(val, str):
            return False, f"ERROR: Argument '{k}' for {tool_name} must be a string, got {type(val).__name__}."
        elif expected_type == int:
            if not isinstance(val, int) or isinstance(val, bool):
                return False, f"ERROR: Argument '{k}' for {tool_name} must be an integer, got {type(val).__name__}."

    # Enums check
    enums = sig.get("enums", {})
    for k, val in args.items():
        if k in enums:
            allowed_vals = enums[k]
            if val not in allowed_vals:
                return (
                    False,
                    f"ERROR: Invalid value '{val}' for argument '{k}'. Allowed values: {', '.join(allowed_vals)}",
                )

    # Value ranges
    if tool_name == "set_alarm_at":
        hour = args.get("hour")
        minute = args.get("minute")
        if hour is not None and not (0 <= hour <= 23):
            return False, f"ERROR: 'hour' must be between 0 and 23, got {hour}."
        if minute is not None and not (0 <= minute <= 59):
            return False, f"ERROR: 'minute' must be between 0 and 59, got {minute}."
    elif tool_name == "set_timer":
        seconds = args.get("seconds")
        if seconds is not None and seconds < 0:
            return False, f"ERROR: 'seconds' must be non-negative, got {seconds}."

    return True, None


def get_tool_schemas():
    if config.BENCHMARK_MODE or not config.USE_MEMORY:
        return [t for t in TOOL_SCHEMAS if t["name"] not in MEMORY_TOOL_NAMES]
    return TOOL_SCHEMAS


def build_mock():
    """A MockTools covering every tool the LLM can see in benchmark mode."""
    return MockTools([t["name"] for t in get_tool_schemas()])


def _send_whatsapp_wrapper(message: str, contact: Optional[str] = None, recipient: Optional[str] = None, **kwargs) -> str:
    target = contact or recipient or "contact"
    return f"WhatsApp message sent to {target}."


def _open_app_wrapper(app_name: str) -> str:
    """Invokes system_tools.open_app and returns real error string if unknown app."""
    return system_tools.open_app(app_name)


def build_dispatch_table(speak_fn=None, mock=None):
    """
    Returns {tool_name: callable}.
    If a MockTools is passed, every tool is replaced by a recording stub,
    except open_app which returns the real error string for unknown apps.
    """
    if mock is not None:
        table = mock.table()
        # Ensure parity: all 17 tools are dispatched
        if "open_app" in table:
            orig_open = table["open_app"]

            def mock_open_app(app_name: str) -> str:
                # Real error check for unknown apps
                res = system_tools.open_app(app_name)
                if res == "App not found on system." or res.startswith("Couldn't open"):
                    if hasattr(mock, "_record"):
                        mock._record("open_app", {"app_name": app_name})
                    return res
                return orig_open(app_name)

            table["open_app"] = mock_open_app

        if "manage_browser" not in table:
            table["manage_browser"] = lambda action, query=None: (
                mock._record("manage_browser", {"action": action, "query": query})
                if hasattr(mock, "_record")
                else None
            ) or f"Browser action done: {action}."
        if "play_pause_media" not in table:
            table["play_pause_media"] = lambda: (
                mock._record("play_pause_media", {}) if hasattr(mock, "_record") else None
            ) or "Toggled media playback."
        if "close_current_app" not in table:
            table["close_current_app"] = lambda: (
                mock._record("close_current_app", {}) if hasattr(mock, "_record") else None
            ) or "Closed active application."
        if "get_window_status" not in table:
            table["get_window_status"] = lambda: (
                mock._record("get_window_status", {}) if hasattr(mock, "_record") else None
            ) or "Active window: Visual Studio Code. Open apps: Chrome, Terminal"
        if "send_whatsapp_message" not in table:
            table["send_whatsapp_message"] = lambda message, contact=None, recipient=None: (
                mock._record("send_whatsapp_message", {"message": message, "contact": contact or recipient})
                if hasattr(mock, "_record")
                else None
            ) or f"WhatsApp message sent to {contact or recipient}."

        return table

    return {
        "remember_note": lambda note: memory_tools.remember_note(note),
        "open_app": _open_app_wrapper,
        "manage_browser": lambda action, query=None: system_tools.manage_browser(action, query),
        "play_pause_media": lambda: system_tools.play_pause_media(),
        "close_current_app": lambda: system_tools.close_current_app(),
        "get_window_status": lambda: system_tools.get_window_status(),
        "send_whatsapp_message": _send_whatsapp_wrapper,
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
