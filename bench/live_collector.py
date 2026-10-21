"""
Live dataset collector for J.A.R.V.I.S. benchmarks.
Captures live spoken audio, transcribes it with Whisper, routes it through
the Ollama model to generate candidate tool and arguments, and logs the interaction
to bench/live_dataset.csv for manual verification.

Usage:
  python -m bench.live_collector
  python -m bench.live_collector --speaker S1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audio import stt
from bench.record import AudioRecorder, save_wav
from core.router import route_command

CSV_HEADERS = [
    "id",
    "category",
    "history",
    "text",
    "expected_tool",
    "expected_args",
    "record_by",
    "audio_path",
]


def ensure_csv_initialized(csv_path: str) -> None:
    """Ensure destination CSV exists and contains the appropriate header row."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def append_live_record(csv_path: str, record: Dict[str, Any]) -> None:
    """Append a validated record to bench/live_dataset.csv."""
    ensure_csv_initialized(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(record)


def process_audio_interaction(
    pcm_bytes: bytes,
    speaker: str,
    audio_dir: str,
    csv_path: str,
    cmd_uuid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Saves raw audio, transcribes with Whisper, passes to Ollama router,
    and logs record to CSV.
    """
    if not pcm_bytes:
        print("[!] No audio captured.")
        return None

    uid = cmd_uuid or uuid.uuid4().hex[:8]
    row_id = f"live_{uid}"
    audio_filename = f"live_cmd_{uid}.wav"
    audio_abs_path = os.path.join(audio_dir, audio_filename)
    audio_rel_path = os.path.join("bench", "audio", audio_filename).replace("\\", "/")

    # 1. Save raw audio
    save_wav(audio_abs_path, pcm_bytes)

    # 2. Transcribe via Faster-Whisper pipeline
    print("Transcribing audio with Faster-Whisper...")
    text = stt.transcribe(audio_abs_path).strip()
    if not text:
        print("[!] No speech recognized in audio.")
        return None

    # 3. Pass through Ollama router to get predicted tool and args
    print("Routing query through Ollama model...")
    routing_res = route_command(text, provider="auto", use_mock=True)

    tool_called = routing_res.tool_called or ""
    tool_args = routing_res.args or {}
    args_json = json.dumps(tool_args)

    # 4. Feedback to speaker in terminal
    print("\n" + "=" * 68)
    print(f"  Command ID       : {row_id}")
    print(f"  Speaker          : {speaker}")
    print(f"  Transcribed Text : \"{text}\"")
    print(f"  Predicted Tool   : {tool_called or 'None (Conversational)'}")
    print(f"  Predicted Args   : {args_json}")
    print(f"  Assistant Reply  : \"{routing_res.reply}\"")
    print(f"  Saved Audio      : {audio_rel_path}")
    print("=" * 68 + "\n")

    # 5. Build and log row
    record = {
        "id": row_id,
        "category": "live_collection",
        "history": "[]",
        "text": text,
        "expected_tool": tool_called,
        "expected_args": args_json,
        "record_by": speaker,
        "audio_path": audio_rel_path,
    }
    append_live_record(csv_path, record)
    return record


def run_live_collector(
    speaker: str,
    audio_dir: str,
    csv_path: str,
    auto_test_mode: bool = False,
) -> None:
    os.makedirs(audio_dir, exist_ok=True)
    ensure_csv_initialized(csv_path)

    print("\n" + "=" * 68)
    print("   J.A.R.V.I.S. - Live Dataset Collector")
    print("=" * 68)
    print(f"  Speaker Identifier : {speaker}")
    print(f"  Audio Output Dir   : {audio_dir}")
    print(f"  Dataset CSV Path   : {csv_path}")
    print("=" * 68)
    print("Instructions:")
    print("  1. Press [Enter] to start speaking, and [Enter] again to stop.")
    print("  2. Type 'q' and press [Enter] at the prompt to quit.")
    print("=" * 68 + "\n")

    if auto_test_mode:
        # Synthetic sine wave audio for automated test runs
        import numpy as np
        synthetic_samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)) * 8000).astype(np.int16)
        process_audio_interaction(
            pcm_bytes=synthetic_samples.tobytes(),
            speaker=speaker,
            audio_dir=audio_dir,
            csv_path=csv_path,
            cmd_uuid="test0001",
        )
        return

    recorder = AudioRecorder()

    while True:
        try:
            choice = input("Press [Enter] to start recording (or 'q' to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting live collector.")
            break

        if choice in ("q", "quit", "exit"):
            print("Session complete. Exiting.")
            break

        print("\n● RECORDING... Speak your request clearly. Press [Enter] when done.")
        recorder.start()
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            recorder.stop()
            break
        pcm_bytes = recorder.stop()

        process_audio_interaction(
            pcm_bytes=pcm_bytes,
            speaker=speaker,
            audio_dir=audio_dir,
            csv_path=csv_path,
        )


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Live Interaction Dataset Collector")
    parser.add_argument("--speaker", type=str, default=None, help="Speaker identifier (e.g. S1, Alice)")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=os.path.join(ROOT, "bench", "audio"),
        help="Directory to save raw WAV files",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(ROOT, "bench", "live_dataset.csv"),
        help="Path to live dataset CSV",
    )
    parser.add_argument(
        "--auto-test",
        action="store_true",
        help="Run non-interactively for automated test verification",
    )
    args = parser.parse_args()

    speaker = args.speaker
    if not speaker:
        speaker = input("Enter speaker identifier (e.g. S1): ").strip()
        if not speaker:
            speaker = "S1"

    run_live_collector(
        speaker=speaker,
        audio_dir=args.audio_dir,
        csv_path=args.csv,
        auto_test_mode=args.auto_test,
    )


if __name__ == "__main__":
    main()
