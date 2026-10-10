"""
Decides what to do with a transcribed voice command.

Simple, high-confidence patterns (open app, add/list/complete todo,
timers) are handled directly without involving an LLM at all — this
keeps common actions fast and free. Anything else, or anything that
explicitly names a model ("ask claude...", "jarvis activate gemini"),
goes to the chosen LLM, which has the same tools available via
function-calling so it can still take actions, not just talk.
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import system_tools, todo_tools, alarm_tools
from tools.tool_registry import build_dispatch_table

from datetime import datetime

MODEL_NAME_PATTERN = re.compile(
    r"\b(?:ask|activate|use|switch to)\s+(claude|chatgpt|gpt|gemini|openai)\b", re.I
)

OPEN_APP_PATTERN = re.compile(r"\b(?:open|launch|start|run|play)\s+(.+)", re.I)
ADD_TODO_PATTERN = re.compile(
    r"\badd\s+(?:a\s+)?(?:task|todo|to[- ]do)\s*(?:to\s+)?(?:my\s+list\s+)?(?:to\s+)?(.+)",
    re.I,
)
ADD_TODO_ALT_PATTERN = re.compile(
    r"\badd\s+(.+?)\s+to\s+(?:my\s+)?(?:to[- ]?do\s*(?:list)?|tasks?\s*(?:list)?)\b",
    re.I,
)
LIST_TODO_PATTERN = re.compile(
    r"\b(?:list|what(?:\x27s|\s+is)\s+on|show)\s+(?:my\s+)?(?:to[- ]?dos?|tasks?)(?:\s+list)?\b",
    re.I,
)
COMPLETE_TODO_PATTERN = re.compile(
    r"\b(?:mark|check off|complete|finish)\s+(.+?)\s+(?:as\s+)?(?:done|complete)\b", re.I
)
TIMER_PATTERN = re.compile(
    r"\b(?:set\s+(?:a\s+)?timer\s+(?:for\s+)?|timer\s+for\s+)(\d+)\s*(minute|min|m|second|sec|s)?",
    re.I,
)
TIME_PATTERN = re.compile(
    r"\b(?:what(?:\x27s|\s+is)\s+(?:the\s+)?time|what\s+time\s+is\s+it|current\s+time)\b",
    re.I,
)
DATE_PATTERN = re.compile(
    r"\b(?:what(?:\x27s|\s+is)\s+(?:the\s+date|today\x27s\s+date)|what\s+day\s+is\s+(?:it|today))\b",
    re.I,
)
SCREEN_PATTERN = re.compile(r"\b(?:screen|what.*(?:looking at|on my screen))\b", re.I)


def extract_requested_model(text: str):
    match = MODEL_NAME_PATTERN.search(text)
    return match.group(1) if match else None


def handle_command(text: str, model_manager, speak_fn):
    """
    Returns the text to speak back to the user.
    """
    dispatch_table = build_dispatch_table(speak_fn)

    # 1. Direct, no-LLM-needed patterns first (fast + free).
    # Time and Date
    if TIME_PATTERN.search(text):
        return f"It is currently {datetime.now().strftime('%I:%M %p')}."

    if DATE_PATTERN.search(text):
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."

    # App Launcher
    m = OPEN_APP_PATTERN.search(text)
    if m and "screen" not in text.lower():
        target = m.group(1).strip()
        target = re.sub(r"[.?!,]+$", "", target).strip()
        target = re.sub(r"\bplease\b", "", target, flags=re.I).strip()
        target = re.sub(r"^(?:up|the|a|an)\s+", "", target, flags=re.I).strip()
        target = re.sub(r"\s+app$", "", target, flags=re.I).strip()
        target = re.sub(r"^(?:up|the|a|an)\s+", "", target, flags=re.I).strip()
        return system_tools.open_app(target)

    # To-Do Management
    m = ADD_TODO_PATTERN.search(text) or ADD_TODO_ALT_PATTERN.search(text)
    if m:
        item = m.group(1).strip().rstrip(".?!")
        return todo_tools.add_todo(item)

    if LIST_TODO_PATTERN.search(text):
        return todo_tools.list_todos()

    m = COMPLETE_TODO_PATTERN.search(text)
    if m:
        item = m.group(1).strip().rstrip(".?!")
        return todo_tools.complete_todo(item)

    # Timer
    m = TIMER_PATTERN.search(text)
    if m:
        val = int(m.group(1))
        unit = (m.group(2) or "minute").lower()
        seconds = val * 60 if unit.startswith("m") else val
        return alarm_tools.set_timer(seconds, speak_fn=speak_fn)

    # 2. Screen-reading question -> take a screenshot, hand to the LLM with the image.
    image_path = None
    if SCREEN_PATTERN.search(text):
        try:
            image_path = system_tools.take_screenshot()
        except RuntimeError as e:
            return str(e)

    # 3. Everything else (including explicit "ask claude..." requests) -> LLM.
    requested_model = extract_requested_model(text)
    client = model_manager.resolve(requested_model)
    if client is None:
        return (
            "I don't have any AI models configured yet — add an API key "
            "to config.py to enable that."
        )

    return client.ask(text, dispatch_table, image_path=image_path)
