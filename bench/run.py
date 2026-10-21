"""
Benchmark Runner for J.A.R.V.I.S.
Usage:
  python -m bench.run --experiment e2 --machine A --provider <name> --out results/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["JARVIS_BENCHMARK"] = "1"

import config
config.BENCHMARK_MODE = True
config.USE_MEMORY = False
config.LLM_TEMPERATURE = 0.0

from bench.dataset import DatasetRow, load_dataset
from bench.env_capture import capture_environment
from bench.mock_tools import BenchmarkMockTools
from bench.providers import get_provider
from bench.scoring import score_call
from bench.telemetry import get_telemetry_snapshot

CSV_COLUMNS = [
    "timestamp",
    "experiment",
    "machine",
    "provider",
    "model",
    "quant",
    "run_number",
    "command_id",
    "category",
    "split",
    "text",
    "expected_tool",
    "expected_args",
    "tool_called",
    "args_called",
    "reply",
    "ttft_ms",
    "total_ms",
    "tokens_in",
    "tokens_out",
    "iterations",
    "tool_ok",
    "args_ok",
    "valid_json",
    "false_tool_call",
    "unsupported_ok",
    "confirmation_ok",
    "gpu_temp_before",
    "gpu_temp_after",
    "gpu_clock_before",
    "gpu_clock_after",
    "gpu_mem_used_before",
    "gpu_mem_used_after",
    "ram_used_pct_before",
    "ram_used_pct_after",
    "error",
]


def load_existing_completed_runs(csv_path: str) -> Set[Tuple[str, int]]:
    completed: Set[Tuple[str, int]] = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cmd_id = row.get("command_id")
                    run_num = row.get("run_number")
                    if cmd_id and run_num and run_num.isdigit():
                        completed.add((cmd_id, int(run_num)))
        except Exception:
            pass
    return completed


def run_benchmark(
    experiment: str,
    machine: str,
    provider_name: str,
    out_dir: str,
    dataset_path: str,
    repeats: int = 5,
    model_tag: str | None = None,
    warmup: int = 1,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    csv_filename = f"{experiment}_{provider_name}_{machine}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    env_json_path = os.path.join(out_dir, "env.json")

    # Capture environment metadata and hashes
    capture_environment(dataset_path=dataset_path, output_path=env_json_path)

    # Load dataset
    dataset: List[DatasetRow] = load_dataset(dataset_path)
    if not dataset:
        raise ValueError(f"No records found in dataset: {dataset_path}")

    # Initialize Provider and Mock Tools
    provider = get_provider(provider_name, model_tag=model_tag)
    mock = BenchmarkMockTools()
    dispatch_table = mock.table()

    # Load already finished (command_id, run_number) for resuming
    completed_runs = load_existing_completed_runs(csv_path)

    # Prepare CSV file
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    print(f"\n============================================================")
    print(f"  J.A.R.V.I.S. Benchmark Runner")
    print(f"============================================================")
    print(f"  Experiment : {experiment}")
    print(f"  Machine    : {machine}")
    print(f"  Provider   : {provider_name} ({provider.model})")
    print(f"  Dataset    : {dataset_path} ({len(dataset)} rows)")
    print(f"  Repeats    : {repeats}")
    print(f"  Output CSV : {csv_path}")
    print(f"  Resumed    : {len(completed_runs)} rows already completed")
    print(f"============================================================\n")

    # Warm-up phase
    if warmup > 0 and len(dataset) > 0:
        print("Executing warm-up run...")
        warmup_item = dataset[0]
        try:
            mock.reset()
            provider.ask(warmup_item.text, dispatch_table, history=warmup_item.history)
        except Exception as e:
            print(f"Warm-up exception (proceeding): {e}")

    # Execution Loop
    for run_num in range(1, repeats + 1):
        # Randomized execution order per repeat
        order_indices = list(range(len(dataset)))
        random.shuffle(order_indices)

        for idx in order_indices:
            row = dataset[idx]
            if (row.id, run_num) in completed_runs:
                continue

            mock.reset()
            t_before = get_telemetry_snapshot()
            timestamp = datetime.datetime.now().isoformat()

            error_msg = None
            try:
                res = provider.ask(row.text, dispatch_table, history=row.history)
            except Exception as ex:
                error_msg = str(ex)
                res = {
                    "reply": f"Fatal execution error: {ex}",
                    "tool_called": None,
                    "args": {},
                    "valid_json": False,
                    "ttft_ms": 0.0,
                    "total_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "iterations": 0,
                    "model": provider.model,
                    "quant": provider.quant,
                    "error": error_msg,
                }

            t_after = get_telemetry_snapshot()

            # Tool execution calls
            executed_calls = mock.calls
            if not executed_calls and res.get("tool_called"):
                executed_calls = [{"tool": res["tool_called"], "args": res.get("args") or {}}]

            # Evaluate scoring
            score = score_call(
                expected_tool=row.expected_tool,
                expected_args=row.expected_args,
                calls=executed_calls,
                valid_json=res.get("valid_json", False),
                reply=res.get("reply", ""),
                category=row.category,
                history=row.history,
            )

            record: Dict[str, Any] = {
                "timestamp": timestamp,
                "experiment": experiment,
                "machine": machine,
                "provider": provider_name,
                "model": res.get("model") or provider.model,
                "quant": res.get("quant") or provider.quant,
                "run_number": run_num,
                "command_id": row.id,
                "category": row.category,
                "split": row.split,
                "text": row.text,
                "expected_tool": row.expected_tool or "",
                "expected_args": json.dumps(row.expected_args),
                "tool_called": res.get("tool_called") or "",
                "args_called": json.dumps(res.get("args") or {}),
                "reply": res.get("reply", ""),
                "ttft_ms": res.get("ttft_ms", 0.0),
                "total_ms": res.get("total_ms", 0.0),
                "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0),
                "iterations": res.get("iterations", 1),
                "tool_ok": score["tool_ok"],
                "args_ok": score["args_ok"],
                "valid_json": score["valid_json"],
                "false_tool_call": score["false_tool_call"],
                "unsupported_ok": score["unsupported_ok"],
                "confirmation_ok": score["confirmation_ok"],
                "gpu_temp_before": t_before["gpu_temp"],
                "gpu_temp_after": t_after["gpu_temp"],
                "gpu_clock_before": t_before["gpu_clock"],
                "gpu_clock_after": t_after["gpu_clock"],
                "gpu_mem_used_before": t_before["gpu_mem_used_mb"],
                "gpu_mem_used_after": t_after["gpu_mem_used_mb"],
                "ram_used_pct_before": t_before["ram_used_pct"],
                "ram_used_pct_after": t_after["ram_used_pct"],
                "error": error_msg or res.get("error") or "",
            }

            writer.writerow(record)
            csv_file.flush()
            completed_runs.add((row.id, run_num))

            status_mark = "OK" if score["tool_ok"] and score["args_ok"] else "FAIL"
            print(
                f"[{status_mark:4}] Run {run_num}/{repeats} | ID: {row.id:3} | {row.text[:30]:30} "
                f"| Tool: {str(record['tool_called']):15} | TTFT: {record['ttft_ms']:6.1f}ms | Total: {record['total_ms']:6.1f}ms"
            )

    csv_file.close()
    print(f"\nBenchmark completed successfully. Results saved to: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Benchmark Suite Runner")
    parser.add_argument("--experiment", type=str, default="e2", help="Experiment name (e.g. e2)")
    parser.add_argument("--machine", type=str, default="A", help="Machine identifier (e.g. A)")
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help="Provider: ollama, regex, stub, or cloud (claude/openai/gemini)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model tag override (for ollama or cloud)")
    parser.add_argument("--out", type=str, default="results/", help="Output directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl"),
        help="Path to dataset JSONL file",
    )
    parser.add_argument("--repeats", "-n", type=int, default=5, help="Number of repetitions per item (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up iterations (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="Run a single-repeat dry run on sample dataset")
    args = parser.parse_args()

    reps = 1 if args.dry_run else args.repeats
    run_benchmark(
        experiment=args.experiment,
        machine=args.machine,
        provider_name=args.provider,
        out_dir=args.out,
        dataset_path=args.dataset,
        repeats=reps,
        model_tag=args.model,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
