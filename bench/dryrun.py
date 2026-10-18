"""
Gate 2 dry run (blueprint Phase 2): push the sample commands through both
strategies with MOCK tools and write a log in the routing CSV schema.

  python bench/dryrun.py

Needs no API keys: the regex path always runs, the LLM path only runs if a
model is configured. Nothing real gets executed (no apps, no to-do edits).
"""

import csv
import datetime
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["JARVIS_BENCHMARK"] = "1"

import config  # noqa: E402

config.BENCHMARK_MODE = True
config.LLM_TEMPERATURE = 0.0

from bench.scoring import score_call  # noqa: E402
from core.router import route  # noqa: E402
from models.model_manager import ModelManager  # noqa: E402
from tools.tool_registry import build_mock  # noqa: E402

COLUMNS = ["timestamp", "strategy", "model_id", "command_id", "category", "split", "run_number",
           "path", "latency_router_ms", "latency_llm_ms", "total_latency_ms", "tokens_in",
           "tokens_out", "cost_usd", "tool_expected", "tool_predicted", "tool_correct",
           "args_correct", "args_lenient", "iterations", "error"]


def load_prices():
    p = os.path.join(ROOT, "bench", "prices.json")
    return json.load(open(p)).get("models", {}) if os.path.exists(p) else {}


def cost(prices, model, tin, tout):
    pr = prices.get(model or "", {})
    if pr.get("in_per_mtok") is None or pr.get("out_per_mtok") is None:
        return ""
    return round(tin * pr["in_per_mtok"] / 1e6 + tout * pr["out_per_mtok"] / 1e6, 6)


def snapshot():
    out = {}
    for p in (config.TODOS_FILE, config.NOTES_FILE):
        out[p] = open(p, "rb").read() if os.path.exists(p) else None
    return out


def main():
    commands = json.load(open(os.path.join(ROOT, "bench", "sample_commands.json")))
    mm = ModelManager()
    mock = build_mock()
    table = mock.table()
    prices = load_prices()
    have_llm = bool(mm.candidates(None))
    before = snapshot()

    print("Direct LLM routing (all commands to LLM):")
    for c in commands:
        print(f"  {c['id']:4} {c['text'][:44]:44} -> llm")

    rows = []
    for strategy in ("hybrid", "always_llm"):
        if strategy == "always_llm" and not have_llm:
            print("\n(always_llm skipped: no model configured)")
            continue
        for c in commands:
            mock.reset()
            t0 = time.perf_counter()
            res = route(c["text"], mm, table, force_llm=(strategy == "always_llm"),
                        capture_screen=False, image_path=c.get("image"))
            total = (time.perf_counter() - t0) * 1000.0
            if res.path == "none" and res.error == "no_model":
                continue  # hybrid command needed an LLM and none is configured
            sc = score_call(c["expected_tool"], c["expected_args"], mock.calls)
            tin, tout = res.stats.get("tokens_in", 0), res.stats.get("tokens_out", 0)
            rows.append({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "strategy": strategy, "model_id": res.model or "", "command_id": c["id"],
                "category": c["category"], "split": c["split"], "run_number": 1,
                "path": res.path, "latency_router_ms": round(res.router_ms, 3),
                "latency_llm_ms": round(res.llm_ms, 1), "total_latency_ms": round(total, 1),
                "tokens_in": tin, "tokens_out": tout, "cost_usd": cost(prices, res.model, tin, tout),
                "tool_expected": c["expected_tool"] or "", "tool_predicted": sc["tool_predicted"] or "",
                "tool_correct": sc["tool_correct"], "args_correct": sc["args_correct"],
                "args_lenient": sc["args_lenient"], "iterations": res.stats.get("iterations", 0),
                "error": res.error or ""})

    out_dir = os.path.join(ROOT, "bench", "out")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "dryrun_log.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    side_effects_clean = snapshot() == before
    print(f"\nWrote {len(rows)} rows to {out}")
    print("No real side effects:", "OK" if side_effects_clean else "FAILED (real files changed!)")
    return 0 if side_effects_clean else 1


if __name__ == "__main__":
    sys.exit(main())
