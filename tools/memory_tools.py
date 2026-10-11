import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import memory


def remember_note(note: str) -> str:
    return memory.append_note(note)
