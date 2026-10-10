"""
Defines the tools Jarvis exposes, both as:
  1. a dispatch table (name -> python function) for direct execution, and
  2. JSON schemas in Anthropic's tool-use format, so an LLM can decide
     to call them itself when a request doesn't match a simple pattern.

Adding a new capability later = write the function, add one entry here.
"""

from tools import system_tools, todo_tools, alarm_tools, gmail_tools, calendar_tools, drive_tools

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
        "description": "Create a Google Calendar event. Convert any spoken relative time (e.g. 'tomorrow at 3pm') into ISO 8601 datetimes before calling this.",
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
]


def build_dispatch_table(speak_fn):
    """
    Returns {tool_name: callable} where alarm tools are pre-wired with
    speak_fn so they can announce themselves out loud when they fire.
    """
    return {
        "open_app": lambda app_name: system_tools.open_app(app_name),
        "add_todo": lambda task: todo_tools.add_todo(task),
        "list_todos": lambda **_: todo_tools.list_todos(),
        "complete_todo": lambda task_hint: todo_tools.complete_todo(task_hint),
        "set_timer": lambda seconds, label="": alarm_tools.set_timer(seconds, label, speak_fn),
        "set_alarm_at": lambda hour, minute, label="": alarm_tools.set_alarm_at(
            hour, minute, label, speak_fn
        ),
        "list_recent_emails": lambda count=5: gmail_tools.list_recent_emails(count),
        "create_email_draft": lambda to, subject, body: gmail_tools.create_email_draft(to, subject, body),
        "list_upcoming_events": lambda count=5: calendar_tools.list_upcoming_events(count),
        "create_event": lambda summary, start_time, end_time, description="": calendar_tools.create_event(
            summary, start_time, end_time, description
        ),
        "list_recent_files": lambda count=5: drive_tools.list_recent_files(count),
    }
