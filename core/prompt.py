import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core import memory


def build_system_prompt() -> str:
    """
    Identity first, then (optionally) remembered notes. In benchmark mode
    the notes are left out so every run sees the same prompt and token count.
    """
    parts = [memory.load_identity()]
    if config.USE_MEMORY and not config.BENCHMARK_MODE:
        notes = memory.load_notes()
        if notes:
            parts.append("Things the user asked you to remember:\n" + notes)
    return "\n\n".join(parts)
