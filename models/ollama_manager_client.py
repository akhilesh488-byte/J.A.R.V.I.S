"""
Fine-tuned Local Ollama Manager Client.
Supports JSON-based single/multi-turn tool calling, validation against tool_registry,
mock execution for benchmarks, and optional escalation cascade to cloud models.
"""

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.prompt import USER_PROFILE_BLOCK, build_system_prompt, load_prompt_templates
from tools import system_tools
from tools.mock_tools import MockTools
from tools.tool_registry import build_dispatch_table, validate_tool_call

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

FIXED_APOLOGY = "I apologize, but I am unable to process that request."


def is_confirmation_question(text: Optional[str]) -> bool:
    """Checks if text is an assistant confirmation question."""
    if not text:
        return False
    lower = text.lower().strip()
    confirm_phrases = [
        "shall i send",
        "should i send",
        "would you like me to send",
        "do you want me to send",
        "send it?",
        "shall i go ahead",
        "should i proceed",
        "are you sure you want to send",
        "confirm sending",
        "ready to send",
    ]
    if any(p in lower for p in confirm_phrases):
        return True
    if "?" in lower and any(w in lower for w in ["send", "whatsapp", "message"]):
        return True
    return False


def is_clear_yes(text: Optional[str]) -> bool:
    """Checks if text is a clear user affirmative confirmation."""
    if not text:
        return False
    cleaned = text.strip().lower().rstrip(".!?,;:")
    yes_exact = {
        "yes",
        "yeah",
        "yep",
        "sure",
        "confirm",
        "confirmed",
        "please do",
        "send it",
        "go ahead",
        "yes please",
        "yes, please",
        "yes please send it",
        "yes, please send it now",
        "yes send it",
        "yes, send it",
        "absolutely",
        "definitely",
        "proceed",
    }
    if cleaned in yes_exact:
        return True
    if cleaned.startswith(("yes", "yeah", "yep", "sure", "please send", "go ahead")):
        return True
    return False


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
        return load_prompt_templates()

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", headers={"User-Agent": "Jarvis"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _build_system_prompt(self) -> str:
        return build_system_prompt(template=self.system_prompt_template, tool_lines=self.tool_lines)

    def _call_ollama(self, messages: List[Dict[str, str]]) -> Tuple[dict, float]:
        options = {
            "temperature": 0.0,
            "num_ctx": 4096,
            "seed": 42,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "options": options,
        }
        logger.info("Ollama call options: %s, keep_alive: -1", options)
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
            resp_json = json.loads(body)

            load_dur_ms = resp_json.get("load_duration", 0) / 1e6
            p_eval_count = resp_json.get("prompt_eval_count", 0)
            p_eval_dur_ms = resp_json.get("prompt_eval_duration", 0) / 1e6
            e_count = resp_json.get("eval_count", 0)
            e_dur_ms = resp_json.get("eval_duration", 0) / 1e6
            logger.info(
                "Ollama timing: load_duration=%.2fms, prompt_eval_count=%d, prompt_eval_duration=%.2fms, eval_count=%d, eval_duration=%.2fms",
                load_dur_ms, p_eval_count, p_eval_dur_ms, e_count, e_dur_ms,
            )
            return resp_json, latency_ms

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
        Enforces strict dispatching, Raw-JSON guarding, and confirmation safety gates.
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
        raw_json_retries = 0

        for iteration in range(4):
            try:
                resp_json, lat_ms = self._call_ollama(messages)
            except Exception as e:
                # If tool execution or gate had set final_reply, honor it on runner completion/error
                if tool_called == "send_whatsapp_message" and final_reply and "User confirmation required" in final_reply:
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
                if error_msg is not None:
                    # In single-turn mocked unit tests or network failure after invalid format
                    return {
                        "reply": final_reply if (final_reply and not final_reply.startswith("{")) else FIXED_APOLOGY,
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
                error_msg = str(e)
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
                # Attempt to parse as JSON tool call
                parse_ok = False
                parsed = None
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        parse_ok = True
                except Exception as ex:
                    parse_ok = False
                    error_msg = f"Invalid JSON tool response: {ex}"

                if not parse_ok or not isinstance(parsed, dict):
                    # Raw-JSON guard: not a valid registered call
                    valid_json = False
                    if error_msg is None:
                        error_msg = "Invalid JSON tool response: missing 'tool' field"
                    err_msg = f"ERROR: {error_msg}"
                    final_reply = FIXED_APOLOGY
                    if raw_json_retries < 1:
                        raw_json_retries += 1
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"TOOL_RESULT: {err_msg}"})
                        continue
                    else:
                        break

                name = parsed.get("tool")
                raw_args = parsed.get("args") or {}
                if not isinstance(raw_args, dict):
                    raw_args = {}

                # Strict validation
                is_valid, v_err = validate_tool_call(name, raw_args)
                if not is_valid:
                    valid_json = True
                    tool_called = name
                    tool_args = raw_args
                    error_msg = v_err
                    final_reply = FIXED_APOLOGY
                    # Raw-JSON guard & strict dispatcher: return 'ERROR: <reason>' as TOOL_RESULT
                    err_msg = v_err if v_err and v_err.startswith("ERROR:") else f"ERROR: {v_err}"
                    if raw_json_retries < 1:
                        raw_json_retries += 1
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"TOOL_RESULT: {err_msg}"})
                        continue
                    else:
                        break

                # Tool call is strictly valid
                valid_json = True
                tool_called = name
                tool_args = raw_args

                # Execute tool
                t_exec_0 = time.perf_counter()
                try:
                    if tool_called == "send_whatsapp_message":
                        # Safety Gate: require prior confirmation question and user clear yes
                        last_assistant_msg = None
                        for turn in reversed(effective_history):
                            if turn.get("role") == "assistant":
                                last_assistant_msg = turn.get("content", "")
                                break

                        has_confirm_question = is_confirmation_question(last_assistant_msg)
                        user_confirmed = is_clear_yes(user_input)

                        # Check if preceded by confirmation or 'to whom' prompt
                        if last_assistant_msg and "to whom" in last_assistant_msg.lower():
                            gate_passed = True
                        elif has_confirm_question and user_confirmed:
                            gate_passed = True
                        else:
                            gate_passed = False

                        if not gate_passed:
                            tool_output = "Error: User confirmation required before sending WhatsApp messages."
                            error_msg = tool_output
                        else:
                            if dispatch_table is not None and tool_called in dispatch_table:
                                tool_output = str(dispatch_table[tool_called](**tool_args))
                            elif force_mock:
                                target = tool_args.get("contact") or tool_args.get("recipient")
                                tool_output = f"WhatsApp message sent to {target}."
                            else:
                                dispatch = build_dispatch_table(speak_fn=None)
                                if tool_called in dispatch:
                                    tool_output = str(dispatch[tool_called](**tool_args))
                                else:
                                    tool_output = f"Executed {tool_called}"
                    elif tool_called == "open_app":
                        # open_app returns real error string for unknown apps
                        app_name = tool_args.get("app_name", "")
                        real_app_res = system_tools.open_app(app_name)
                        if real_app_res == "App not found on system." or real_app_res.startswith("Couldn't open"):
                            tool_output = real_app_res
                        elif dispatch_table is not None and tool_called in dispatch_table:
                            tool_output = str(dispatch_table[tool_called](**tool_args))
                        elif force_mock:
                            tool_output = f"Opened {app_name}."
                        else:
                            tool_output = real_app_res
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
                    tool_output = f"ERROR: Error executing tool {tool_called}: {ex}"

                tool_latency_ms += (time.perf_counter() - t_exec_0) * 1000.0

                final_reply = tool_output
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"TOOL_RESULT: {tool_output}"})
            else:
                # Regular spoken/text reply
                final_reply = content
                break

        # Final raw-JSON safety guard: never speak raw JSON
        if final_reply.strip().startswith("{"):
            final_reply = FIXED_APOLOGY

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
