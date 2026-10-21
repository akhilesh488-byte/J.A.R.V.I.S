"""
Scoring rules for benchmarking Jarvis.
Evaluates model tool execution against benchmark dataset targets:
1. First tool call only: Only the first tool call in the turn is scored.
2. expected_tool null: Expects no tool call. If a tool was called, false_tool_call=True.
3. Argument comparison: Checks ONLY the arguments listed in expected_args, normalized (lowercase, stripped).
4. Unsupported-request rule: Final reply must not claim success for impossible/unsupported requests.
5. Confirmation rule: send_whatsapp_message must be preceded by a confirmation question from the assistant.

Outputs:
- tool_ok: bool
- args_ok: bool
- valid_json: bool
- false_tool_call: bool
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

UNSUPPORTED_SUCCESS_PATTERNS = [
    re.compile(r"\b(?:i have|i've|here is your|ordered your|made the|baked the|washing|cooked)\b", re.I),
    re.compile(r"\b(?:success(?:fully)?|done!|completed your order)\b", re.I),
]

CONFIRMATION_PATTERNS = [
    re.compile(r"\b(?:shall i send|confirm|to whom|should i send|would you like me to send)\b", re.I),
    re.compile(r"\b(?:send it|send this message)\?\b", re.I),
]


def _norm(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        # Normalize numeric representations
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return str(val)
    return str(val).strip().lower()


def check_unsupported_reply(reply: str) -> bool:
    """
    Returns True if the reply does NOT falsely claim success for an unsupported task.
    """
    if not reply:
        return True
    reply_lower = reply.strip().lower()
    for pat in UNSUPPORTED_SUCCESS_PATTERNS:
        if pat.search(reply_lower):
            # If it says "I cannot... here is why", ensure it's not a negated phrase
            if any(neg in reply_lower for neg in ["cannot", "can't", "unable", "not able", "afraid i can't", "sorry"]):
                continue
            return False
    return True


def check_whatsapp_confirmation(history: Optional[List[Dict[str, str]]]) -> bool:
    """
    Returns True if the preceding conversation history includes a confirmation
    question from the assistant before executing send_whatsapp_message.
    """
    if not history:
        return False

    for turn in reversed(history):
        if turn.get("role") == "assistant":
            content = turn.get("content", "").strip()
            for pat in CONFIRMATION_PATTERNS:
                if pat.search(content):
                    return True
            if "?" in content:
                return True
            break
    return False


def score_call(
    expected_tool: Optional[str],
    expected_args: Optional[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    valid_json: bool = True,
    reply: Optional[str] = None,
    category: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Score a single benchmark execution turn.
    """
    first_call = calls[0] if calls else None
    pred_tool = first_call["tool"] if first_call else None
    pred_args = first_call["args"] if first_call else {}

    # Rule 1 & 2: Expected tool null vs predicted tool
    if expected_tool is None:
        if calls:
            false_tool_call = True
            tool_ok = False
            args_ok = False
        else:
            false_tool_call = False
            tool_ok = True
            args_ok = True

        # Rule 4: Unsupported request check
        if category in ("unsupported", "adversarial") and reply:
            if not check_unsupported_reply(reply):
                tool_ok = False
                args_ok = False

        return {
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "valid_json": valid_json if calls else True,
            "false_tool_call": false_tool_call,
            "tool_predicted": pred_tool,
            "tool_correct": tool_ok,
            "args_correct": args_ok,
            "args_lenient": args_ok,
            "unsupported_ok": check_unsupported_reply(reply or ""),
            "confirmation_ok": True,
        }

    # Expected tool is not null
    false_tool_call = False
    if not calls:
        return {
            "tool_ok": False,
            "args_ok": False,
            "valid_json": valid_json,
            "false_tool_call": False,
            "tool_predicted": None,
            "tool_correct": False,
            "args_correct": False,
            "args_lenient": False,
            "unsupported_ok": True,
            "confirmation_ok": True,
        }

    tool_ok = (pred_tool == expected_tool)

    # Rule 5: WhatsApp confirmation rule
    confirmation_ok = True
    if pred_tool == "send_whatsapp_message":
        confirmation_ok = check_whatsapp_confirmation(history)
        if not confirmation_ok:
            tool_ok = False

    # Rule 3: Compare only expected args, normalized lowercase and stripped
    args_ok = False
    args_lenient = False
    if tool_ok:
        exp = {k: _norm(v) for k, v in (expected_args or {}).items()}
        pred = {k: _norm(v) for k, v in pred_args.items()}
        args_ok = all(pred.get(k) == v for k, v in exp.items())
        args_lenient = all(k in pred and (v in pred[k] or pred[k] in v) for k, v in exp.items())

    return {
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "valid_json": valid_json,
        "false_tool_call": false_tool_call,
        "tool_predicted": pred_tool,
        "tool_correct": tool_ok,
        "args_correct": args_ok,
        "args_lenient": args_lenient,
        "unsupported_ok": True,
        "confirmation_ok": confirmation_ok,
    }
