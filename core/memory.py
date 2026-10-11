"""
Plain-markdown memory: an identity file and a notes file. No database.
The identity file is loaded into every system prompt; the notes file is
appended to by the remember_note tool and its tail is loaded too.
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DEFAULT_IDENTITY = (
    "You are Jarvis, a friendly voice assistant on the user's laptop. "
    "Replies are spoken aloud: keep them short and plain, no markdown."
)


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def load_identity() -> str:
    return _read(config.IDENTITY_FILE) or DEFAULT_IDENTITY


def load_notes(max_chars: int = 3000) -> str:
    text = _read(config.NOTES_FILE)
    return text[-max_chars:] if len(text) > max_chars else text


def append_note(note: str) -> str:
    os.makedirs(os.path.dirname(config.NOTES_FILE), exist_ok=True)
    line = f"- {datetime.date.today().isoformat()}: {note.strip()}\n"
    with open(config.NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    return "Noted."
