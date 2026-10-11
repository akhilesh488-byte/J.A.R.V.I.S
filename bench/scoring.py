"""
Scoring rules. DECIDE THESE BEFORE RUNNING ANYTHING and don't change them
after you've seen results:

- Only the FIRST tool call is scored.
- expected_tool = None means "no tool should be called".
- Only the arguments listed in expected_args are checked; extra arguments
  the model adds (e.g. label="") are ignored.
- Values are compared lowercase and stripped.
- 'args_correct' is strict equality; 'args_lenient' also accepts one value
  containing the other (e.g. "buy groceries" vs "buy groceries to my list").
Report which one the paper uses.
"""


def _norm(v):
    return str(v).strip().lower()


def score_call(expected_tool, expected_args, calls):
    """calls = [{'tool': name, 'args': {...}}, ...] in call order."""
    predicted = calls[0]["tool"] if calls else None

    if expected_tool is None:
        ok = len(calls) == 0
        return {"tool_predicted": predicted, "tool_correct": ok,
                "args_correct": ok, "args_lenient": ok}

    if not calls:
        return {"tool_predicted": None, "tool_correct": False,
                "args_correct": False, "args_lenient": False}

    first = calls[0]
    tool_ok = first["tool"] == expected_tool
    pred = {k: _norm(v) for k, v in first["args"].items()}
    exp = {k: _norm(v) for k, v in (expected_args or {}).items()}
    strict = tool_ok and all(pred.get(k) == v for k, v in exp.items())
    lenient = tool_ok and all(k in pred and (v in pred[k] or pred[k] in v) for k, v in exp.items())
    return {"tool_predicted": predicted, "tool_correct": tool_ok,
            "args_correct": strict, "args_lenient": lenient}
