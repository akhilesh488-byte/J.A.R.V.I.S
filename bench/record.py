"""
Audio recording tool for the J.A.R.V.I.S. benchmark test set.
Records raw 16 kHz mono WAV directly from the default microphone.
Usage:
  python -m bench.record --speaker S1 --csv jarvis_test_set_template.csv --out bench/audio/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import threading
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.dataset import parse_csv_template

RATE = 16000
CHANNELS = 1
CHUNK = 1024
SAMPLE_WIDTH = 2  # 16-bit PCM

LOG_COLUMNS = ["id", "speaker", "path", "duration_s", "peak_level", "clipped", "timestamp"]


def get_audio_filename(cmd_id: str, speaker: str) -> str:
    """Generate audio filename formatted as <id>_<speaker>.wav."""
    return f"{cmd_id}_{speaker}.wav"


def get_audio_path(out_dir: str, cmd_id: str, speaker: str) -> str:
    """Generate full audio file path."""
    return os.path.join(out_dir, get_audio_filename(cmd_id, speaker))


def analyze_audio(pcm_bytes: bytes, rate: int = RATE) -> Dict[str, Any]:
    """
    Analyze raw int16 PCM audio samples for peak level, clipping, and duration.
    """
    if not pcm_bytes:
        return {
            "duration_s": 0.0,
            "peak_level": 0,
            "clipped": False,
            "is_low": True,
        }

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    duration_s = round(len(samples) / float(rate), 2)
    peak_level = int(np.max(np.abs(samples))) if len(samples) > 0 else 0
    clipped = bool(np.any(np.abs(samples) >= 32767))
    is_low = bool(peak_level < 1000)

    return {
        "duration_s": duration_s,
        "peak_level": peak_level,
        "clipped": clipped,
        "is_low": is_low,
    }


def save_wav(path: str, pcm_bytes: bytes, rate: int = RATE) -> None:
    """Save raw 16 kHz mono WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)


def log_recording(
    log_csv_path: str,
    cmd_id: str,
    speaker: str,
    audio_path: str,
    duration_s: float,
    peak_level: int,
    clipped: bool,
    timestamp: Optional[str] = None,
) -> None:
    """
    Append record to recordings.csv with schema:
    id, speaker, path, duration_s, peak_level, clipped, timestamp
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_csv_path)), exist_ok=True)
    file_exists = os.path.exists(log_csv_path) and os.path.getsize(log_csv_path) > 0

    ts = timestamp or datetime.datetime.now().isoformat()
    record = {
        "id": cmd_id,
        "speaker": speaker,
        "path": audio_path,
        "duration_s": duration_s,
        "peak_level": peak_level,
        "clipped": clipped,
        "timestamp": ts,
    }

    with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def load_commands_for_speaker(csv_path: str, speaker: str, noisy: bool = False) -> List[Dict[str, str]]:
    """
    Load matching commands from jarvis_test_set_template.csv where record_by matches speaker.
    If noisy=True, limits to the first 30 IDs only.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Template CSV not found: {csv_path}")

    speaker_clean = speaker.strip().upper()
    matching: List[Dict[str, str]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd_id = (row.get("id") or "").strip()
            if not cmd_id:
                continue

            record_by = (row.get("record_by") or "").strip().upper()
            # Split record_by on ';' or ','
            speakers = [s.strip() for s in record_by.replace(",", ";").split(";") if s.strip()]

            if speaker_clean in speakers or not record_by:
                text = (row.get("text") or "").strip()
                if not text:
                    text = (row.get("hint") or "").strip()
                if not text and row.get("target_tool"):
                    text = f"open {row['target_tool']}" if row["target_tool"] == "open_app" else row["target_tool"]

                matching.append({
                    "id": cmd_id,
                    "text": text,
                    "target_tool": row.get("target_tool", ""),
                    "category": row.get("category", ""),
                })

    if noisy:
        matching = matching[:30]

    return matching


class AudioRecorder:
    def __init__(self, rate: int = RATE, channels: int = CHANNELS, chunk: int = CHUNK):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self._frames: List[bytes] = []
        self._is_recording = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        import pyaudio

        self._frames = []
        self._is_recording = True

        def _record_thread():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
            while self._is_recording:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    self._frames.append(data)
                except Exception:
                    break
            stream.stop_stream()
            stream.close()
            p.terminate()

        self._thread = threading.Thread(target=_record_thread, daemon=True)
        self._thread.start()

    def stop(self) -> bytes:
        self._is_recording = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return b"".join(self._frames)


def run_recording_session(
    speaker: str,
    csv_path: str,
    out_dir: str,
    noisy: bool = False,
    auto_test_mode: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_csv_path = os.path.join(out_dir, "recordings.csv")

    commands = load_commands_for_speaker(csv_path, speaker, noisy=noisy)
    if not commands:
        print(f"No commands found for speaker '{speaker}' in {csv_path}")
        return

    print("=" * 68)
    print(f"  J.A.R.V.I.S. Benchmark Audio Capture Tool")
    print(f"  Speaker  : {speaker}")
    print(f"  Dataset  : {csv_path}")
    print(f"  Out Dir  : {out_dir}")
    print(f"  Mode     : {'NOISY (First 30 IDs)' if noisy else 'STANDARD'}")
    print(f"  Commands : {len(commands)}")
    print("=" * 68)
    print("Controls: Enter to Start/Stop | 'r' to Re-record | 's' to Skip | 'q' to Quit\n")

    recorder = AudioRecorder()

    for idx, cmd in enumerate(commands, start=1):
        cmd_id = cmd["id"]
        text = cmd["text"]
        target_path = get_audio_path(out_dir, cmd_id, speaker)

        # Resume logic: skip existing recordings
        if os.path.exists(target_path):
            print(f"[{idx}/{len(commands)}] Skipping existing: {cmd_id} ({os.path.basename(target_path)})")
            continue

        print("-" * 68)
        print(f"[{idx}/{len(commands)}] Command ID: {cmd_id}")
        print(f"Text to speak:\n  \"{text}\"")
        print("-" * 68)

        if auto_test_mode:
            # Synthetic 1-second sine wave for headless / automated runs
            synthetic_samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, RATE)) * 10000).astype(np.int16)
            pcm_bytes = synthetic_samples.tobytes()
            analysis = analyze_audio(pcm_bytes)
            save_wav(target_path, pcm_bytes)
            log_recording(
                log_csv_path,
                cmd_id,
                speaker,
                target_path,
                analysis["duration_s"],
                analysis["peak_level"],
                analysis["clipped"],
            )
            print(f"Saved: {target_path}")
            continue

        while True:
            action = input("Press [Enter] to start recording ('s' to skip, 'q' to quit): ").strip().lower()
            if action == "q":
                print("\nRecording session terminated by user. Good day, sir.")
                return
            if action == "s":
                print(f"Skipped {cmd_id}.")
                break

            # Start recording
            print("\n[RECORDING...] Speak clearly into your mic. Press [Enter] to stop.")
            recorder.start()
            input()
            pcm_bytes = recorder.stop()

            analysis = analyze_audio(pcm_bytes)
            duration_s = analysis["duration_s"]
            peak = analysis["peak_level"]
            clipped = analysis["clipped"]
            is_low = analysis["is_low"]

            print(f"Captured: {duration_s:.2f}s | Peak: {peak}/32767 | Clipped: {clipped}")

            # Warnings
            if clipped:
                print(">> [WARNING] Audio CLIPPED! Peak level hit full scale. Please lower your microphone input.")
            elif is_low:
                print(">> [WARNING] Audio level very LOW! Speak closer or increase microphone volume.")

            review = input("Accept recording? [Enter] to Save, 'r' to Re-record, 's' to Skip, 'q' to Quit: ").strip().lower()
            if review == "q":
                print("\nRecording session terminated by user.")
                return
            if review == "s":
                print(f"Discarded and skipped {cmd_id}.")
                break
            if review == "r":
                print("Re-recording...")
                continue

            # Save recording
            save_wav(target_path, pcm_bytes)
            log_recording(
                log_csv_path,
                cmd_id=cmd_id,
                speaker=speaker,
                audio_path=target_path,
                duration_s=duration_s,
                peak_level=peak,
                clipped=clipped,
            )
            print(f">> Saved to {target_path}\n")
            break

    print("\nAll target audio commands for this speaker have been recorded!")


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Benchmark Audio Recorder")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker identifier (e.g. S1, S2, S3, S4)")
    parser.add_argument("--csv", type=str, default="jarvis_test_set_template.csv", help="Path to template CSV")
    parser.add_argument("--out", type=str, default=None, help="Output directory for recordings")
    parser.add_argument("--noisy", action="store_true", help="Record noisy audio set (writes to bench/audio_noisy/, first 30 IDs only)")
    parser.add_argument("--auto-test", action="store_true", help="Run in automated test mode without interactive mic input")
    args = parser.parse_args()

    out_directory = args.out
    if not out_directory:
        out_directory = os.path.join(ROOT, "bench", "audio_noisy" if args.noisy else "audio")

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(ROOT, csv_path)

    run_recording_session(
        speaker=args.speaker,
        csv_path=csv_path,
        out_dir=out_directory,
        noisy=args.noisy,
        auto_test_mode=args.auto_test,
    )


if __name__ == "__main__":
    main()
