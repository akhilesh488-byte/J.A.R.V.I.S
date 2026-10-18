import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core import memory

USER_PROFILE_BLOCK = """About the user (ambient knowledge: use it naturally, never recite it unprompted):
- Name: Akhilesh Masali. Address him as Akhilesh, sir, or Mr. Masali; mostly sir.
- Age: 20.
- Studies: final-year B.E. student in Artificial Intelligence and Machine Learning (AIML) at PDA College of Engineering, Kalaburagi, Karnataka.
- Tech: builds LangGraph and MCP workflows on an Asus TUF Gaming F15 that runs Linux and Windows 11.
- Rides: a Bajaj Pulsar N160.
- Interests: high-protein gym routines; the games Red Dead Redemption 2 and The Last of Us; the shows Suits and The Mentalist."""


def build_system_prompt() -> str:
    """
    Identity first, user profile block, then (optionally) remembered notes.
    In benchmark mode the notes are left out so every run sees the same prompt and token count.
    """
    parts = [memory.load_identity(), USER_PROFILE_BLOCK.strip()]
    if config.USE_MEMORY and not config.BENCHMARK_MODE:
        notes = memory.load_notes()
        if notes:
            parts.append("Things the user asked you to remember:\n" + notes)
    return "\n\n".join(parts)

