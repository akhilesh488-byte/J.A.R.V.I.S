import hashlib
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core import memory

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are J.A.R.V.I.S., an efficient AI assistant for desktop control and tasks.
Current Time: {timestamp}

{PROFILE}

You have access to the following tools:
{tool_lines}

If you need to call a tool, respond ONLY with a JSON object in this exact format:
{"tool": "<tool_name>", "args": {<argument_name>: <value>}}
Do NOT wrap the JSON in markdown code fences.

If no tool is needed or after receiving TOOL_RESULT, respond naturally and concisely to the user.
"""

USER_PROFILE_BLOCK = """About the user (ambient knowledge: use it naturally, never recite it unprompted):
- Name: Akhilesh Masali. Address him as Akhilesh, sir, or Mr. Masali; mostly sir.
- Age: 20.
- Studies: final-year B.E. student in Artificial Intelligence and Machine Learning (AIML) at PDA College of Engineering, Kalaburagi, Karnataka.
- Tech: builds LangGraph and MCP workflows on an Asus TUF Gaming F15 that runs Linux and Windows 11.
- Rides: a Bajaj Pulsar N160.
- Interests: high-protein gym routines; the games Red Dead Redemption 2 and The Last of Us; the shows Suits and The Mentalist."""

DEFAULT_TOOL_LINES = """open_app(app_name: str) - Open an application
manage_browser(action: str, query: str = None) - Manage browser actions, searching, or opening URLs
play_pause_media() - Toggle media playback
close_current_app(window_title: str = None) - Close active application
get_window_status() - Get active window and open applications
send_whatsapp_message(recipient: str, message: str) - Send a WhatsApp message
add_todo(task: str) - Add a task to to-do list
list_todos() - List current to-do items
complete_todo(task_hint: str) - Mark to-do item as done
set_timer(seconds: int, label: str = None) - Set a countdown timer
set_alarm_at(hour: int, minute: int, label: str = None) - Set an alarm
list_recent_emails(count: int = None) - List recent emails
create_email_draft(to: str, subject: str, body: str) - Create email draft
list_upcoming_events(count: int = None) - List calendar events
create_event(summary: str, start_time: str, end_time: str, description: str = None) - Create calendar event
list_recent_files(count: int = None) - List recent files
remember_note(note: str) - Remember a note
"""


def load_prompt_templates() -> Tuple[str, str]:
    """Loads system_prompt_template.txt and tool_lines.txt from disk or defaults."""
    candidate_dirs = [
        os.path.join(ROOT, "jarvis_data"),
        ROOT,
        os.path.join(ROOT, "data"),
    ]
    template_str = None
    tool_lines_str = None
    for d in candidate_dirs:
        t_path = os.path.join(d, "system_prompt_template.txt")
        tl_path = os.path.join(d, "tool_lines.txt")
        if template_str is None and os.path.exists(t_path):
            try:
                with open(t_path, "r", encoding="utf-8") as f:
                    template_str = f.read()
            except Exception:
                pass
        if tool_lines_str is None and os.path.exists(tl_path):
            try:
                with open(tl_path, "r", encoding="utf-8") as f:
                    tool_lines_str = f.read()
            except Exception:
                pass
    return (
        template_str if template_str is not None else DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        tool_lines_str if tool_lines_str is not None else DEFAULT_TOOL_LINES,
    )


def estimate_tokens(text: str) -> int:
    """Estimates token count for prompt logging."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text.split())


def build_system_prompt(
    timestamp: Optional[str] = None,
    template: Optional[str] = None,
    tool_lines: Optional[str] = None,
) -> str:
    """
    Builds the system prompt exactly from template:
    - Identity & prompt structure from template
    - Current date/time line
    - User profile block (kept even in benchmark mode)
    - tool_lines.txt verbatim
    - Only remembered notes memory may be disabled in benchmark mode.
    Logs token count and SHA-256 hash of final prompt.
    """
    tpl, tools = load_prompt_templates()
    if template is not None:
        tpl = template
    if tool_lines is not None:
        tools = tool_lines

    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = tpl.replace("{timestamp}", ts)
    prompt = prompt.replace("{tool_lines}", tools.strip())
    prompt = prompt.replace("{tools}", tools.strip())

    if "{PROFILE}" in prompt:
        prompt = prompt.replace("{PROFILE}", USER_PROFILE_BLOCK.strip())
    elif "{profile}" in prompt:
        prompt = prompt.replace("{profile}", USER_PROFILE_BLOCK.strip())
    else:
        prompt = prompt + "\n\n" + USER_PROFILE_BLOCK.strip()

    # Custom identity support if defined in memory
    try:
        id_text = memory.load_identity()
        if id_text and id_text != memory.DEFAULT_IDENTITY and memory.DEFAULT_IDENTITY in prompt:
            prompt = prompt.replace(memory.DEFAULT_IDENTITY, id_text)
    except Exception:
        pass

    # Only notes memory may be disabled in benchmark mode
    if config.USE_MEMORY and not config.BENCHMARK_MODE:
        try:
            notes = memory.load_notes()
            if notes:
                prompt += f"\n\nThings the user asked you to remember:\n{notes}"
        except Exception:
            pass

    sha256_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    token_cnt = estimate_tokens(prompt)
    logger.info("System prompt built: %d tokens, SHA-256: %s", token_cnt, sha256_hash)

    return prompt
