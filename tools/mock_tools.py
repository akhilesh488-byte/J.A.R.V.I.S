"""
Mock tools for benchmarking and testing.
Implements exact typed signatures for all standard tools, records (tool_name, args, timestamp)
to an in-memory list, and returns realistic string responses without side effects.
"""

import time
from typing import Any, Dict, List, Optional, Tuple


class ToolCallRecord(tuple):
    """
    Represents a recorded tool call (tool_name, args, timestamp).
    Supports tuple indexing/unpacking and dictionary-style access.
    """
    def __new__(cls, tool_name: str, args: dict, timestamp: float):
        return super().__new__(cls, (tool_name, args, timestamp))

    @property
    def tool(self) -> str:
        return self[0]

    @property
    def args(self) -> dict:
        return self[1]

    @property
    def timestamp(self) -> float:
        return self[2]

    @property
    def t(self) -> float:
        return self[2]

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("tool", "name"):
                return self[0]
            if item == "args":
                return self[1]
            if item in ("timestamp", "t"):
                return self[2]
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class MockTools:
    def __init__(self, names: Optional[List[str]] = None):
        self.names = list(names) if names is not None else [
            "open_app",
            "add_todo",
            "list_todos",
            "complete_todo",
            "set_timer",
            "set_alarm_at",
            "list_recent_emails",
            "create_email_draft",
            "list_upcoming_events",
            "create_event",
            "list_recent_files",
            "remember_note",
        ]
        self.calls: List[ToolCallRecord] = []

    def reset(self):
        self.calls.clear()

    def _record(self, tool_name: str, args: dict) -> float:
        ts = time.time()
        record = ToolCallRecord(tool_name, args, ts)
        self.calls.append(record)
        return ts

    def open_app(self, app_name: str) -> str:
        self._record("open_app", {"app_name": app_name})
        return f"Opened {app_name}."

    def add_todo(self, task: str) -> str:
        self._record("add_todo", {"task": task})
        return f"Added task: {task}"

    def list_todos(self) -> str:
        self._record("list_todos", {})
        return "1. Buy groceries\n2. Call dentist\n3. Review pull request"

    def complete_todo(self, task_hint: str) -> str:
        self._record("complete_todo", {"task_hint": task_hint})
        return f"Completed todo matching '{task_hint}'."

    def set_timer(self, seconds: int, label: str = None) -> str:
        args = {"seconds": seconds}
        if label is not None:
            args["label"] = label
        self._record("set_timer", args)
        lbl = f" ({label})" if label else ""
        return f"Timer set for {seconds} seconds{lbl}."

    def set_alarm_at(self, hour: int, minute: int, label: str = None) -> str:
        args = {"hour": hour, "minute": minute}
        if label is not None:
            args["label"] = label
        self._record("set_alarm_at", args)
        lbl = f" ({label})" if label else ""
        return f"Alarm set for {hour:02d}:{minute:02d}{lbl}."

    def list_recent_emails(self, count: int = None) -> str:
        c = count if count is not None else 5
        self._record("list_recent_emails", {"count": count} if count is not None else {})
        return f"Found {c} recent emails: 1. Team Update, 2. Invoice #1024, 3. Weekly Digest."

    def create_email_draft(self, to: str, subject: str, body: str) -> str:
        self._record("create_email_draft", {"to": to, "subject": subject, "body": body})
        return f"Drafted email to {to} with subject '{subject}'."

    def list_upcoming_events(self, count: int = None) -> str:
        c = count if count is not None else 5
        self._record("list_upcoming_events", {"count": count} if count is not None else {})
        return f"Found {c} upcoming events: 1. Sprint Planning (10:00 AM), 2. Lunch with Sarah (12:30 PM)."

    def create_event(self, summary: str, start_time: str, end_time: str, description: str = None) -> str:
        args = {"summary": summary, "start_time": start_time, "end_time": end_time}
        if description is not None:
            args["description"] = description
        self._record("create_event", args)
        return f"Created event '{summary}' from {start_time} to {end_time}."

    def list_recent_files(self, count: int = None) -> str:
        c = count if count is not None else 5
        self._record("list_recent_files", {"count": count} if count is not None else {})
        return f"Found {c} recent files: 1. document.pdf, 2. project_notes.docx, 3. data.csv."

    def remember_note(self, note: str) -> str:
        self._record("remember_note", {"note": note})
        return f"Remembered: {note}"

    def table(self) -> Dict[str, Any]:
        mapping = {
            "open_app": self.open_app,
            "add_todo": self.add_todo,
            "list_todos": self.list_todos,
            "complete_todo": self.complete_todo,
            "set_timer": self.set_timer,
            "set_alarm_at": self.set_alarm_at,
            "list_recent_emails": self.list_recent_emails,
            "create_email_draft": self.create_email_draft,
            "list_upcoming_events": self.list_upcoming_events,
            "create_event": self.create_event,
            "list_recent_files": self.list_recent_files,
            "remember_note": self.remember_note,
        }
        return {k: v for k, v in mapping.items() if k in self.names}
