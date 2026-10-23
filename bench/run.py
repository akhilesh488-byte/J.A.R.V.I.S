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
from bench.env_capture import capture_environment, compare_runs, get_heavy_running_processes
from bench.mock_tools import BenchmarkMockTools
from bench.providers import get_provider, run_ollama_diagnostic
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
    "audio_path",
    "stt_ms",
    "transcribed_text",
    "text",
    "history_used",
    "expected_tool",
    "expected_args",
    "expected_behavior",
    "tool_called",
    "args_called",
    "reply",
    "ttft_ms",
    "load_duration_ms",
    "prompt_eval_count",
    "prompt_eval_duration_ms",
    "eval_count",
    "eval_duration_ms",
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


def build_audio_index(audio_dir: Optional[str]) -> Dict[str, str]:
    """
    Build index mapping command_id -> full audio WAV path.
    Checks recordings.csv first, then scans directory for {id}_*.wav or {id}.wav.
    """
    if not audio_dir:
        return {}
    target_dir = audio_dir if os.path.isabs(audio_dir) else os.path.join(ROOT, audio_dir)
    if not os.path.exists(target_dir):
        return {}

    index: Dict[str, str] = {}
    # 1. recordings.csv
    rec_csv = os.path.join(target_dir, "recordings.csv")
    if os.path.exists(rec_csv):
        try:
            with open(rec_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = (row.get("id") or "").strip()
                    p = (row.get("path") or "").strip()
                    if cid and p:
                        full_p = p if os.path.isabs(p) else os.path.join(ROOT, p)
                        if os.path.exists(full_p):
                            index[cid] = full_p
        except Exception:
            pass

    # 2. Directory scan for any file starting with <id>_ or matching <id>.wav
    try:
        for fname in os.listdir(target_dir):
            if fname.lower().endswith(".wav"):
                base_id = fname.split("_")[0].split(".")[0]
                if base_id not in index:
                    index[base_id] = os.path.join(target_dir, fname)
    except Exception:
        pass

    return index


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
    dataset_path: str | None = None,
    repeats: int = 5,
    model_tag: str | None = None,
    warmup: int = 1,
    audio_dir: str | None = None,
    split: str | None = None,
    csv_filename: str | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if not csv_filename:
        split_suffix = f"_{split}" if (split and not (experiment.endswith(f"_{split}") or experiment == split)) else ""
        csv_filename = f"{experiment}{split_suffix}_{provider_name}_{machine}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    env_json_path = os.path.join(out_dir, "env.json")

    # Resolve dataset_path if not provided
    if not dataset_path:
        if split == "dev":
            dev_jsonl = os.path.join(ROOT, "bench", "data", "dev_set.jsonl")
            dataset_path = dev_jsonl if os.path.exists(dev_jsonl) else os.path.join(ROOT, "jarvis_test_set_template.csv")
        elif split == "test":
            test_jsonl = os.path.join(ROOT, "bench", "data", "test_set.jsonl")
            dataset_path = test_jsonl if os.path.exists(test_jsonl) else os.path.join(ROOT, "jarvis_test_set_template.csv")
        else:
            default_template = os.path.join(ROOT, "jarvis_test_set_template.csv")
            dataset_path = default_template if os.path.exists(default_template) else os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")

    # Capture environment metadata and hashes
    capture_environment(dataset_path=dataset_path, output_path=env_json_path)

    # Load dataset with strict split filtering
    dataset: List[DatasetRow] = load_dataset(dataset_path, split=split)
    if not dataset:
        split_info = f" matching split '{split}'" if split else ""
        raise ValueError(f"No records found in dataset: {dataset_path}{split_info}")

    # Build audio index if audio_dir provided
    audio_index = build_audio_index(audio_dir)
    if audio_dir:
        # Strictly filter dataset to only rows with genuine recorded audio WAV files
        original_count = len(dataset)
        dataset = [r for r in dataset if r.id in audio_index]
        if not dataset:
            split_info = f" (split='{split}')" if split else ""
            raise ValueError(f"No audio recordings found in {audio_dir} matching dataset {dataset_path}{split_info}")
        print(f"  Audio Mode : Strictly evaluating {len(dataset)}/{original_count} commands with raw .wav audio (zero synthetic fallback)")

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
    print(f"  REMINDER: Please close all IDEs (VS Code, etc.) and browsers")
    print(f"  (Chrome, Edge, etc.) before running to minimize resource contention.")
    heavy_procs = get_heavy_running_processes()
    if heavy_procs:
        procs_str = ", ".join(f"{p['name']} (PID {p['pid']})" for p in heavy_procs)
        print(f"  [HYGIENE WARNING] Heavy running processes: {procs_str}")
    print(f"------------------------------------------------------------")
    print(f"  Experiment : {experiment}")
    print(f"  Machine    : {machine}")
    print(f"  Provider   : {provider_name} ({provider.model})")
    print(f"  Split      : {split or 'all'}")
    print(f"  Dataset    : {dataset_path} ({len(dataset)} rows)")
    if audio_dir:
        print(f"  Audio Dir  : {audio_dir} ({len(audio_index)} matching recordings indexed)")
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
            warmup_text = warmup_item.text
            if audio_dir and audio_index.get(warmup_item.id):
                from audio import stt
                warmup_text = stt.transcribe(audio_index[warmup_item.id]).strip() or warmup_item.text
            provider.ask(warmup_text, dispatch_table, history=warmup_item.history)
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

            # Ingest raw .wav inputs and convert via STT (zero synthetic fallback)
            audio_path = audio_index.get(row.id) or ""
            stt_ms = 0.0

            if audio_dir:
                if not audio_path or not os.path.exists(audio_path):
                    continue  # Strict: skip anything without a real audio file
                from audio import stt
                t_stt_0 = time.perf_counter()
                transcription_err = None
                try:
                    transcribed = stt.transcribe(audio_path).strip()
                except Exception as ex:
                    transcribed = ""
                    transcription_err = f"STT error: {ex}"
                stt_ms = (time.perf_counter() - t_stt_0) * 1000.0

                # Strict audio pipeline: use transcribed text directly, zero synthetic fallback
                prompt_text = transcribed
                error_msg = transcription_err
                if not prompt_text and not error_msg:
                    error_msg = f"Empty audio transcription from {audio_path}"
            else:
                prompt_text = row.text
                error_msg = None

            mock.reset()
            t_before = get_telemetry_snapshot()
            timestamp = datetime.datetime.now().isoformat()

            error_msg = None
            try:
                res = provider.ask(prompt_text, dispatch_table, history=row.history)
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
            # History used in production format
            history_used_str = json.dumps(row.history) if row.history else ""

            # Evaluate scoring
            score = score_call(
                expected_tool=row.expected_tool,
                expected_args=row.expected_args,
                calls=executed_calls,
                valid_json=res.get("valid_json", False),
                reply=res.get("reply", ""),
                category=row.category,
                history=row.history,
                has_label=getattr(row, "has_label", None),
                user_prompt=prompt_text,
            )

            has_label = score.get("has_label", True)
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
                "audio_path": os.path.relpath(audio_path, ROOT).replace("\\", "/") if audio_path else "",
                "stt_ms": round(stt_ms, 1),
                "transcribed_text": prompt_text,
                "text": prompt_text,
                "history_used": history_used_str,
                "expected_tool": row.expected_tool or "",
                "expected_args": json.dumps(row.expected_args),
                "expected_behavior": getattr(row, "expected_behavior", "") or getattr(row, "expected_reply_type", "") or "",
                "tool_called": res.get("tool_called") or "",
                "args_called": json.dumps(res.get("args") or {}),
                "reply": res.get("reply", ""),
                "ttft_ms": res.get("ttft_ms", 0.0),
                "load_duration_ms": res.get("load_duration_ms", 0.0),
                "prompt_eval_count": res.get("prompt_eval_count", res.get("tokens_in", 0)),
                "prompt_eval_duration_ms": res.get("prompt_eval_duration_ms", 0.0),
                "eval_count": res.get("eval_count", res.get("tokens_out", 0)),
                "eval_duration_ms": res.get("eval_duration_ms", 0.0),
                "total_ms": round(res.get("total_ms", 0.0) + stt_ms, 2),
                "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0),
                "iterations": res.get("iterations", 1),
                "tool_ok": str(score["tool_ok"]) if (has_label and score["tool_ok"] is not None) else "",
                "args_ok": str(score["args_ok"]) if (has_label and score["args_ok"] is not None) else "",
                "valid_json": str(score["valid_json"]) if (has_label and score["valid_json"] is not None) else ("True" if res.get("valid_json") else "False"),
                "false_tool_call": str(score["false_tool_call"]) if (has_label and score["false_tool_call"] is not None) else "",
                "unsupported_ok": str(score["unsupported_ok"]) if (has_label and score["unsupported_ok"] is not None) else "",
                "confirmation_ok": str(score["confirmation_ok"]) if (has_label and score["confirmation_ok"] is not None) else "",
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
            stt_label = f"STT: {stt_ms:5.0f}ms | " if audio_path else ""
            print(
                f"[{status_mark:4}] Run {run_num}/{repeats} | ID: {row.id:3} | {stt_label}{prompt_text[:30]:30} "
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
        default="stub",
        help="Provider: ollama, regex, stub, or cloud (claude/openai/gemini)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model tag override (for ollama or cloud)")
    parser.add_argument("--split", type=str, choices=["dev", "test"], default=None, help="Benchmark split to run: 'dev' (60 commands) or 'test' (60 commands)")
    parser.add_argument("--csv-name", type=str, default=None, help="Explicit filename for output CSV")
    parser.add_argument("--out", type=str, default="results/", help="Output directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSONL or CSV file (defaults according to --split or dry-run)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Path to directory containing WAV recordings for end-to-end audio benchmark",
    )
    parser.add_argument("--repeats", "-n", type=int, default=5, help="Number of repetitions per item (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up iterations (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="Run a single-repeat dry run on sample dataset")
    parser.add_argument("--rescore", type=str, default=None, help="Rescore an existing benchmark CSV against ground truth")
    parser.add_argument("--diagnostic", action="store_true", help="Run 20-call timing diagnostic on Ollama (load/prefill/decode split + ollama ps)")
    parser.add_argument("--compare", nargs=2, metavar=("RUN1", "RUN2"), help="Compare two benchmark runs by their env.json hashes")
    args = parser.parse_args()

    if args.diagnostic:
        run_ollama_diagnostic(model=args.model)
        return

    if args.compare:
        r1, r2 = args.compare
        try:
            res = compare_runs(r1, r2)
            print(f"\n[OK] Run environments match perfectly: {r1} vs {r2}")
        except ValueError as ex:
            print(f"\n[REFUSED] {ex}")
            sys.exit(1)
        return

    if args.rescore:
        from bench.scoring import rescore_results_csv
        rescored = rescore_results_csv(args.rescore)
        print(f"Successfully rescored {len(rescored)} rows in {args.rescore}")
        return

    dataset_path = args.dataset
    if not dataset_path:
        if args.dry_run:
            dataset_path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
        elif args.split == "dev":
            dataset_path = os.path.join(ROOT, "bench", "data", "dev_set.jsonl")
        elif args.split == "test":
            dataset_path = os.path.join(ROOT, "bench", "data", "test_set.jsonl")
        else:
            dataset_path = os.path.join(ROOT, "jarvis_test_set_template.csv")

    reps = 1 if args.dry_run else args.repeats
    run_benchmark(
        experiment=args.experiment,
        machine=args.machine,
        provider_name=args.provider,
        out_dir=args.out,
        dataset_path=dataset_path,
        repeats=reps,
        model_tag=args.model,
        warmup=args.warmup,
        audio_dir=args.audio_dir,
        split=args.split,
        csv_filename=args.csv_name,
    )


if __name__ == "__main__":
    main()
