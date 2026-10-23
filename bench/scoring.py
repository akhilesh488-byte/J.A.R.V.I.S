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

import csv
import json
import os
import re
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def check_whatsapp_confirmation(
    history: Optional[List[Dict[str, str]]],
    user_input: Optional[str] = None,
) -> bool:
    """
    Returns True if the preceding conversation history includes a confirmation
    question from the assistant and user affirmed before executing send_whatsapp_message.
    """
    if not history:
        return False

    last_assistant = None
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            last_assistant = turn.get("content", "").strip()
            break

    if not last_assistant:
        return False

    has_confirm_q = False
    for pat in CONFIRMATION_PATTERNS:
        if pat.search(last_assistant):
            has_confirm_q = True
            break
    if not has_confirm_q and "?" in last_assistant:
        if any(w in last_assistant.lower() for w in ["send", "whatsapp", "message", "to whom"]):
            has_confirm_q = True

    if not has_confirm_q:
        return False

    if user_input is not None:
        user_clean = user_input.strip().lower()
        if "to whom" in last_assistant.lower():
            return True
        yes_tokens = {"yes", "yeah", "yep", "sure", "confirm", "send", "proceed", "please"}
        return any(tok in user_clean for tok in yes_tokens)

    return True


def score_call(
    expected_tool: Optional[str],
    expected_args: Optional[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    valid_json: bool = True,
    reply: Optional[str] = None,
    category: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    has_label: Optional[bool] = None,
    user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score a single benchmark execution turn.
    Only computes accuracy flags when ground-truth labels exist.
    confirmation_ok and unsupported_ok are real behavioral checks.
    """
    first_call = calls[0] if calls else None
    pred_tool = first_call["tool"] if first_call else None
    pred_args = first_call["args"] if first_call else {}

    # Check if ground-truth label exists
    if has_label is None:
        has_label = bool(
            expected_tool is not None
            or (category and str(category).lower() not in ("", "none", "null", "unlabeled", "unknown", "tool"))
            or (expected_args is not None and len(expected_args) > 0)
            or expected_args == {}
        )

    if not has_label:
        return {
            "has_label": False,
            "tool_ok": None,
            "args_ok": None,
            "valid_json": valid_json if calls else True,
            "false_tool_call": None,
            "tool_predicted": pred_tool,
            "tool_correct": None,
            "args_correct": None,
            "args_lenient": None,
            "unsupported_ok": None,
            "confirmation_ok": None,
        }

    # Expected tool null (e.g. unsupported, clarify, conversational, near-miss)
    if expected_tool is None:
        if calls:
            false_tool_call = True
            tool_ok = False
            args_ok = False
        else:
            false_tool_call = False
            tool_ok = True
            args_ok = True

        # Real unsupported_ok check
        unsupported_ok = True
        if category in ("unsupported", "adversarial"):
            if calls:
                unsupported_ok = False
                tool_ok = False
            else:
                unsupported_ok = check_unsupported_reply(reply or "")
                if not unsupported_ok:
                    tool_ok = False
                    args_ok = False

        return {
            "has_label": True,
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "valid_json": valid_json if calls else True,
            "false_tool_call": false_tool_call,
            "tool_predicted": pred_tool,
            "tool_correct": tool_ok,
            "args_correct": args_ok,
            "args_lenient": args_ok,
            "unsupported_ok": unsupported_ok,
            "confirmation_ok": True,
        }

    # Expected tool is not null
    false_tool_call = False
    if not calls:
        # Check if expected tool was send_whatsapp_message and model correctly asked for confirmation
        confirmation_ok = True
        if expected_tool == "send_whatsapp_message":
            confirmation_ok = check_whatsapp_confirmation(history, user_input=user_prompt)

        return {
            "has_label": True,
            "tool_ok": False,
            "args_ok": False,
            "valid_json": valid_json,
            "false_tool_call": False,
            "tool_predicted": None,
            "tool_correct": False,
            "args_correct": False,
            "args_lenient": False,
            "unsupported_ok": True,
            "confirmation_ok": confirmation_ok,
        }

    tool_ok = (pred_tool == expected_tool)

    # Real confirmation_ok check for WhatsApp
    confirmation_ok = True
    if pred_tool == "send_whatsapp_message" or expected_tool == "send_whatsapp_message":
        confirmation_ok = check_whatsapp_confirmation(history, user_input=user_prompt)
        if pred_tool == "send_whatsapp_message" and not confirmation_ok:
            tool_ok = False

    # Compare arguments
    args_ok = False
    args_lenient = False
    if tool_ok:
        exp = {k: _norm(v) for k, v in (expected_args or {}).items()}
        pred = {k: _norm(v) for k, v in pred_args.items()}

        if "contact" in exp and "contact" not in pred and "recipient" in pred:
            pred["contact"] = pred["recipient"]
        if "recipient" in exp and "recipient" not in pred and "contact" in pred:
            pred["recipient"] = pred["contact"]

        def _match_val(expected_val: str, predicted_val: Optional[str]) -> bool:
            if expected_val.startswith("<") and expected_val.endswith(">"):
                return bool(predicted_val)
            return predicted_val == expected_val

        args_ok = all(_match_val(v, pred.get(k)) for k, v in exp.items())
        args_lenient = all(
            k in pred and (_match_val(v, pred[k]) or v in pred[k] or pred[k] in v)
            for k, v in exp.items()
        )

    return {
        "has_label": True,
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


def rescore_results_csv(
    csv_path: str,
    out_path: Optional[str] = None,
    labels_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rescores an existing results CSV file against dev60 ground-truth labels and returns rows.
    Updates expected_tool, expected_args, split, category, and recalculated accuracy flags:
    tool_ok, args_ok, valid_json, false_tool_call, unsupported_ok, confirmation_ok.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    target_labels = labels_path or os.path.join(ROOT, "dev60_proposed_labels.csv")
    dev_labels: Dict[str, Dict[str, str]] = {}
    if os.path.exists(target_labels):
        with open(target_labels, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("id") or "").strip()
                if cid:
                    dev_labels[cid] = row

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    updated_rows: List[Dict[str, Any]] = []
    for r in rows:
        cid = (r.get("command_id") or "").strip()
        dev = dev_labels.get(cid, {})

        # Ground truth tool
        if "expected_tool" in dev:
            exp_tool = (dev.get("expected_tool") or "").strip()
            if not exp_tool or exp_tool.lower() in ("none", "null"):
                exp_tool = None
        else:
            exp_tool = (r.get("expected_tool") or "").strip() or None

        # Ground truth args
        if "expected_args_json" in dev:
            raw_args = (dev.get("expected_args_json") or "").strip()
            try:
                exp_args = json.loads(raw_args) if raw_args else {}
            except Exception:
                exp_args = {}
        else:
            raw_args = (r.get("expected_args") or "").strip()
            try:
                exp_args = json.loads(raw_args) if raw_args else {}
            except Exception:
                exp_args = {}

        # Split and category
        row_split = "dev" if cid in dev_labels else (r.get("split") or "test")
        category = (dev.get("proposed_category") or r.get("category") or "").strip() or "tool"

        # Model call
        called_tool = (r.get("tool_called") or "").strip() or None
        called_args_raw = (r.get("args_called") or "").strip()
        try:
            called_args = json.loads(called_args_raw) if called_args_raw else {}
        except Exception:
            called_args = {}
        calls = [{"tool": called_tool, "args": called_args}] if called_tool else []

        valid_json = r.get("valid_json", "").lower() in ("true", "1", "yes")
        reply = r.get("reply") or ""

        # History turns
        history: List[Dict[str, str]] = []
        earlier_u = (dev.get("earlier_user") or "").strip()
        earlier_j = (dev.get("earlier_jarvis") or "").strip()
        if earlier_u:
            history.append({"role": "user", "content": earlier_u})
        if earlier_j:
            history.append({"role": "assistant", "content": earlier_j})

        has_label = bool(
            exp_tool is not None
            or (category and category.lower() not in ("", "none", "null", "unlabeled", "unknown", "tool"))
            or row_split == "dev"
        )

        prompt_str = r.get("transcribed_text") or r.get("text") or ""

        score = score_call(
            expected_tool=exp_tool,
            expected_args=exp_args,
            calls=calls,
            valid_json=valid_json,
            reply=reply,
            category=category,
            history=history,
            has_label=has_label,
            user_prompt=prompt_str,
        )

        r["split"] = row_split
        r["category"] = category
        r["expected_tool"] = exp_tool or ""
        r["expected_args"] = json.dumps(exp_args)
        r["transcribed_text"] = prompt_str
        if "text" in r and not r["text"]:
            r["text"] = prompt_str
        if "history_used" not in r:
            r["history_used"] = json.dumps(history) if history else ""
        r["tool_ok"] = str(score["tool_ok"]) if (has_label and score["tool_ok"] is not None) else ""
        r["args_ok"] = str(score["args_ok"]) if (has_label and score["args_ok"] is not None) else ""
        r["valid_json"] = str(score["valid_json"]) if (has_label and score["valid_json"] is not None) else ("True" if valid_json else "False")
        r["false_tool_call"] = str(score["false_tool_call"]) if (has_label and score["false_tool_call"] is not None) else ""
        r["unsupported_ok"] = str(score["unsupported_ok"]) if (has_label and score["unsupported_ok"] is not None) else ""
        r["confirmation_ok"] = str(score["confirmation_ok"]) if (has_label and score["confirmation_ok"] is not None) else ""
        updated_rows.append(r)

    dest = out_path or csv_path
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        # Ensure fieldnames has all keys from updated rows
        out_fields = list(fieldnames)
        for extra in ["transcribed_text", "history_used"]:
            if extra not in out_fields:
                out_fields.append(extra)
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(updated_rows)

    return updated_rows

