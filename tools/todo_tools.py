"""
Simple local to-do list, stored as JSON. No cloud sync in Phase 1 —
that can come later if you want it synced across devices.
"""

import json
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _load() -> list:
    if not os.path.exists(config.TODOS_FILE):
        return []
    with open(config.TODOS_FILE, "r") as f:
        return json.load(f)


def _save(todos: list) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.TODOS_FILE, "w") as f:
        json.dump(todos, f, indent=2)


def add_todo(task: str) -> str:
    todos = _load()
    todos.append({"id": str(uuid.uuid4())[:8], "task": task, "done": False})
    _save(todos)
    return f"Added to your list: {task}"


def list_todos(include_done: bool = False) -> str:
    todos = _load()
    if not include_done:
        todos = [t for t in todos if not t["done"]]
    if not todos:
        return "Your to-do list is empty."
    lines = [f"{i+1}. {t['task']}" for i, t in enumerate(todos)]
    return "Here's your list: " + "; ".join(lines)


def complete_todo(task_hint: str) -> str:
    """
    Marks the first not-done todo whose text contains task_hint
    (case-insensitive) as done. Voice commands are messy, so this is
    a "close enough" match rather than requiring exact text.
    """
    todos = _load()
    hint = task_hint.strip().lower()
    for t in todos:
        if not t["done"] and hint in t["task"].lower():
            t["done"] = True
            _save(todos)
            return f"Marked '{t['task']}' as done."
    return f"Couldn't find an open task matching '{task_hint}'."
