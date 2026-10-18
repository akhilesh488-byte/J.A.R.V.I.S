"""
Fine-tuned Local Ollama Manager Client.
Supports JSON-based single/multi-turn tool calling, validation against tool_registry,
mock execution for benchmarks, and optional escalation cascade to cloud models.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tools import system_tools
from tools.mock_tools import MockTools
from tools.tool_registry import build_dispatch_table, validate_tool_call

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

V2_SYSTEM_TOOLS = {
    "manage_browser",
    "play_pause_media",
    "close_current_app",
    "get_window_status",
    "send_whatsapp_message",
}


class OllamaManagerClient:
    name = "ollama_manager"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        cascade_enabled: bool = False,
        timeout: Optional[float] = None,
        model_manager: Any = None,
    ):
        self.model = model or os.environ.get("LOCAL_LLM_MODEL") or os.environ.get("OLLAMA_MODEL", "jarvis-3b")
        raw_host = host or os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = raw_host.rstrip("/")
        self.cascade_enabled = cascade_enabled
        self.timeout = timeout if timeout is not None else float(os.environ.get("OLLAMA_TIMEOUT", "90.0"))
        self.model_manager = model_manager
        self.last_stats: Dict[str, Any] = {}
        self.chat_history: List[Dict[str, str]] = []

        self.system_prompt_template, self.tool_lines = self._load_templates()

    def clear_history(self) -> None:
        """Clears the conversational chat history."""
        self.chat_history = []

    def _load_templates(self) -> Tuple[str, str]:
        paths_to_check = [
            os.path.join(ROOT, "jarvis_data"),
            os.path.join(os.getcwd(), "jarvis_data"),
            os.path.join(ROOT, "data"),
        ]

        prompt_tpl = None
        tool_lines = None

        for p in paths_to_check:
            tpl_path = os.path.join(p, "system_prompt_template.txt")
            tools_path = os.path.join(p, "tool_lines.txt")
            if prompt_tpl is None and os.path.exists(tpl_path):
                try:
                    with open(tpl_path, "r", encoding="utf-8") as f:
                        prompt_tpl = f.read()
                except Exception:
                    pass
            if tool_lines is None and os.path.exists(tools_path):
                try:
                    with open(tools_path, "r", encoding="utf-8") as f:
                        tool_lines = f.read()
                except Exception:
                    pass

        return (
            prompt_tpl if prompt_tpl is not None else DEFAULT_SYSTEM_PROMPT_TEMPLATE,
            tool_lines if tool_lines is not None else DEFAULT_TOOL_LINES,
        )

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", headers={"User-Agent": "Jarvis"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _build_system_prompt(self) -> str:
        # If files were added to jarvis_data after init, refresh if using default
        if self.system_prompt_template == DEFAULT_SYSTEM_PROMPT_TEMPLATE:
            loaded_tpl, loaded_tools = self._load_templates()
            if loaded_tpl != DEFAULT_SYSTEM_PROMPT_TEMPLATE:
                self.system_prompt_template = loaded_tpl
                self.tool_lines = loaded_tools

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = self.system_prompt_template.replace("{timestamp}", now_str)
        prompt = prompt.replace("{tool_lines}", self.tool_lines.strip())
        prompt = prompt.replace("{tools}", self.tool_lines.strip())

        if "{PROFILE}" in prompt:
            prompt = prompt.replace("{PROFILE}", USER_PROFILE_BLOCK.strip())
        elif "{profile}" in prompt:
            prompt = prompt.replace("{profile}", USER_PROFILE_BLOCK.strip())
        else:
            prompt = prompt + "\n\n" + USER_PROFILE_BLOCK.strip()

        if config.USE_MEMORY and not config.BENCHMARK_MODE:
            try:
                from core import memory
                notes = memory.load_notes()
                if notes:
                    prompt += f"\n\nRemembered Information & Notes:\n{notes}"
                id_text = memory.load_identity()
                if id_text and id_text != memory.DEFAULT_IDENTITY:
                    prompt = id_text + "\n\n" + prompt
            except Exception:
                pass
        return prompt

    def _call_ollama(self, messages: List[Dict[str, str]]) -> Tuple[dict, float]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            body = resp.read().decode("utf-8")
            return json.loads(body), latency_ms

    def _escalate(
        self,
        user_input: str,
        tool_called: Optional[str],
        args: Optional[dict],
        valid_json: bool,
        llm_latency_ms: float,
        tool_latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        force_mock: bool,
    ) -> dict:
        try:
            from models.model_manager import ModelManager
            mgr = self.model_manager or ModelManager()
            cloud_candidates = [
                c for c in mgr.candidates() if getattr(c, "name", "") not in ("local", "ollama_manager")
            ]
            if cloud_candidates:
                client = cloud_candidates[0]
                t_c0 = time.perf_counter()
                if force_mock:
                    dispatch = MockTools().table()
                else:
                    dispatch = build_dispatch_table(speak_fn=None)
                cloud_reply = client.ask(user_input, dispatch)
                cloud_ms = (time.perf_counter() - t_c0) * 1000.0
                stats = getattr(client, "last_stats", {}) or {}
                reply_text = str(cloud_reply)
                u_str = user_input.strip()
                r_str = reply_text.strip()
                if (
                    not u_str.startswith("{")
                    and not u_str.startswith("TOOL_RESULT:")
                    and r_str
                    and not r_str.startswith("{")
                ):
                    self.chat_history.append({"role": "user", "content": user_input})
                    self.chat_history.append({"role": "assistant", "content": reply_text})
                return {
                    "reply": reply_text,
                    "tool_called": tool_called,
                    "args": args,
                    "valid_json": valid_json,
                    "llm_latency_ms": round(llm_latency_ms + cloud_ms, 2),
                    "tool_latency_ms": round(tool_latency_ms, 2),
                    "tokens_in": tokens_in + stats.get("prompt_tokens", 0),
                    "tokens_out": tokens_out + stats.get("completion_tokens", 0),
                    "error": None,
                    "escalated": True,
                }
        except Exception as e:
            return {
                "reply": f"Cascade error: {e}",
                "tool_called": tool_called,
                "args": args,
                "valid_json": valid_json,
                "llm_latency_ms": round(llm_latency_ms, 2),
                "tool_latency_ms": round(tool_latency_ms, 2),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "error": f"Cascade failed: {e}",
                "escalated": False,
            }

        return {
            "reply": "No cloud model configured for escalation.",
            "tool_called": tool_called,
            "args": args,
            "valid_json": valid_json,
            "llm_latency_ms": round(llm_latency_ms, 2),
            "tool_latency_ms": round(tool_latency_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "error": "No cloud provider available",
            "escalated": False,
        }

    def ask(
        self,
        user_input: str,
        dispatch_table: Optional[Any] = None,
        image_path: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        force_mock: bool = False,
        **kwargs,
    ) -> dict:
        """
        Interacts with local Ollama manager model with tool execution and multi-turn loops.
        """
        if isinstance(dispatch_table, list) and history is None:
            history = dispatch_table
            dispatch_table = None

        system_prompt = self._build_system_prompt()
        messages: List[Dict[str, str]] = []

        effective_history = history if history is not None else self.chat_history
        if effective_history:
            for turn in effective_history:
                content = (turn.get("content") or "").strip()
                # Selective Memory: Only standard conversational turns, no JSON or TOOL_RESULT
                if not content.startswith("{") and not content.startswith("TOOL_RESULT:"):
                    messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_input})

        llm_latency_ms = 0.0
        tool_latency_ms = 0.0
        tokens_in = 0
        tokens_out = 0
        tool_called = None
        tool_args = None
        valid_json = False
        final_reply = ""
        error_msg = None
        mock_instance = MockTools() if force_mock else None

        for iteration in range(3):
            try:
                resp_json, lat_ms = self._call_ollama(messages)
            except Exception as e:
                error_msg = str(e)
                if tool_called == "send_whatsapp_message" and final_reply == "Error: User confirmation required before sending WhatsApp messages.":
                    return {
                        "reply": final_reply,
                        "tool_called": tool_called,
                        "args": tool_args,
                        "valid_json": True,
                        "llm_latency_ms": round(llm_latency_ms, 2),
                        "tool_latency_ms": round(tool_latency_ms, 2),
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "error": final_reply,
                        "escalated": False,
                    }
                if self.cascade_enabled:
                    return self._escalate(
                        user_input, tool_called, tool_args, valid_json,
                        llm_latency_ms, tool_latency_ms, tokens_in, tokens_out, force_mock
                    )
                return {
                    "reply": f"Ollama connection error: {e}",
                    "tool_called": tool_called,
                    "args": tool_args,
                    "valid_json": False,
                    "llm_latency_ms": round(llm_latency_ms, 2),
                    "tool_latency_ms": round(tool_latency_ms, 2),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "error": error_msg,
                    "escalated": False,
                }

            llm_latency_ms += lat_ms
            tokens_in += resp_json.get("prompt_eval_count", 0)
            tokens_out += resp_json.get("eval_count", 0)

            content = resp_json.get("message", {}).get("content", "").strip()

            if content.startswith("{"):
                try:
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict) or "tool" not in parsed:
                        raise ValueError("Output started with '{' but is missing required 'tool' field")
                    valid_json = True
                except Exception as e:
                    valid_json = False
                    if self.cascade_enabled:
                        return self._escalate(
                            user_input, tool_called, tool_args, valid_json,
                            llm_latency_ms, tool_latency_ms, tokens_in, tokens_out, force_mock
                        )
                    return {
                        "reply": content,
                        "tool_called": None,
                        "args": None,
                        "valid_json": False,
                        "llm_latency_ms": round(llm_latency_ms, 2),
                        "tool_latency_ms": round(tool_latency_ms, 2),
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "error": f"Invalid JSON tool response: {e}",
                        "escalated": False,
                    }

                name = parsed.get("tool")
                raw_args = parsed.get("args") or {}
                if not isinstance(raw_args, dict):
                    raw_args = {}

                is_valid, v_err = validate_tool_call(name, raw_args)
                if not is_valid and name in V2_SYSTEM_TOOLS:
                    is_valid = True
                    v_err = None

                if not is_valid:
                    if self.cascade_enabled:
                        return self._escalate(
                            user_input, name, raw_args, True,
                            llm_latency_ms, tool_latency_ms, tokens_in, tokens_out, force_mock
                        )
                    return {
                        "reply": f"Tool validation error: {v_err}",
                        "tool_called": name,
                        "args": raw_args,
                        "valid_json": True,
                        "llm_latency_ms": round(llm_latency_ms, 2),
                        "tool_latency_ms": round(tool_latency_ms, 2),
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "error": v_err,
                        "escalated": False,
                    }

                # Valid tool call
                tool_called = name
                tool_args = raw_args

                # Execute tool
                t_exec_0 = time.perf_counter()
                try:
                    if tool_called == "send_whatsapp_message":
                        # WhatsApp Safety Gate: Check chat_history
                        last_assistant_msg = None
                        for h in (self.chat_history, history or []):
                            for turn in reversed(h):
                                if turn.get("role") == "assistant":
                                    last_assistant_msg = turn.get("content", "")
                                    break
                            if last_assistant_msg is not None:
                                break

                        gate_passed = False
                        if last_assistant_msg:
                            if (
                                "Shall I send" in last_assistant_msg
                                or "To whom" in last_assistant_msg
                                or "shall i send" in last_assistant_msg.lower()
                                or "to whom" in last_assistant_msg.lower()
                            ):
                                gate_passed = True

                        if not gate_passed:
                            tool_output = "Error: User confirmation required before sending WhatsApp messages."
                        else:
                            if dispatch_table is not None and tool_called in dispatch_table:
                                tool_output = str(dispatch_table[tool_called](**tool_args))
                            elif force_mock:
                                tool_output = f"Executed {tool_called}"
                            else:
                                dispatch = build_dispatch_table(speak_fn=None)
                                if tool_called in dispatch:
                                    tool_output = str(dispatch[tool_called](**tool_args))
                                else:
                                    tool_output = f"Executed {tool_called}"
                    else:
                        if dispatch_table is not None and tool_called in dispatch_table:
                            tool_output = str(dispatch_table[tool_called](**tool_args))
                        elif force_mock:
                            dispatch = mock_instance.table()
                            if tool_called in dispatch:
                                tool_output = str(dispatch[tool_called](**tool_args))
                            elif hasattr(system_tools, tool_called):
                                tool_output = str(getattr(system_tools, tool_called)(**tool_args))
                            else:
                                tool_output = f"Mock executed {tool_called}"
                        else:
                            dispatch = build_dispatch_table(speak_fn=None)
                            if tool_called in dispatch:
                                tool_output = str(dispatch[tool_called](**tool_args))
                            elif hasattr(system_tools, tool_called):
                                tool_output = str(getattr(system_tools, tool_called)(**tool_args))
                            else:
                                tool_output = f"Executed {tool_called}"
                except Exception as ex:
                    tool_output = f"Error executing tool {tool_called}: {ex}"
                tool_latency_ms += (time.perf_counter() - t_exec_0) * 1000.0

                final_reply = tool_output
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"TOOL_RESULT: {tool_output}"})
            else:
                # Regular spoken/text reply
                final_reply = content
                break

        # Selective Memory Rule: Only append standard conversational turns to chat_history.
        # Do NOT append JSON tool calls (strings starting with '{') or 'TOOL_RESULT:' messages.
        u_stripped = user_input.strip()
        r_stripped = final_reply.strip() if final_reply else ""
        if (
            not u_stripped.startswith("{")
            and not u_stripped.startswith("TOOL_RESULT:")
            and r_stripped
            and not r_stripped.startswith("{")
            and not r_stripped.startswith("Ollama connection error:")
        ):
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": final_reply})

        res = {
            "reply": final_reply,
            "tool_called": tool_called,
            "args": tool_args,
            "valid_json": valid_json,
            "llm_latency_ms": round(llm_latency_ms, 2),
            "tool_latency_ms": round(tool_latency_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "error": error_msg,
            "escalated": False,
        }
        self.last_stats = res
        return res
