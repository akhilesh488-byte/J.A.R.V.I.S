#!/usr/bin/env python
"""
STT Bake-off tool.

Usage:
    python bench/stt_bakeoff.py record          # walk through phrases, record WAVs
    python bench/stt_bakeoff.py run             # run all available engines over recordings
    python bench/stt_bakeoff.py run --engines faster_whisper deepgram

Commands:
    record  Walk through bench/stt_phrases.txt, record each phrase, save to
            bench/audio/<id>.wav and bench/audio/references.json.
            Press R to re-record the current phrase, Enter to accept, S to skip.

    run     Warm up each available engine once, then run every engine over all
            recordings N times (default 1). Writes CSV to bench/stt_results_<ts>.csv.
            Prints per-engine summary: median latency, mean WER, exact-match %.

Normalization (documented here, do not change after first run):
    - Lowercase
    - Strip leading/trailing whitespace
    - Remove punctuation: .,!?;:'"()-
    - Numerals: keep as words (e.g. "five" stays "five"; "5" stays "5" — both
      count correct only if they match the reference exactly)
    - WER computed with jiwer, using the above normalization.

Dependencies:
    jiwer (bench only) — add to requirements-bench.txt
    PyAudio, numpy — already in requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import wave

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, ROOT)

import config  # noqa: E402

PHRASES_FILE = os.path.join(BENCH_DIR, "stt_phrases.txt")
AUDIO_DIR = os.path.join(BENCH_DIR, "audio")
REFERENCES_FILE = os.path.join(AUDIO_DIR, "references.json")

RATE = 16000
CHUNK = 1024


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize for WER/CER computation:
      - lowercase
      - strip punctuation: .,!?;:'"()-
      - collapse whitespace
    Numerals kept as-is (see module docstring for policy).
    """
    import re
    text = text.lower().strip()
    text = re.sub(r"[.,!?;:'\"()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _list_input_devices():
    import pyaudio
    pa = pyaudio.PyAudio()
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) >= 1:
            devices.append((i, info["name"]))
    pa.terminate()
    return devices


def record_phrase(save_path: str, duration_s: float = 5.0) -> str:
    """
    Record up to duration_s of audio from the microphone.
    Uses energy VAD from stt.py to stop early.
    Returns save_path.
    """
    import pyaudio
    from audio.vad import EnergyEndpointer

    pa = pyaudio.PyAudio()
    open_kwargs = dict(format=pyaudio.paInt16, channels=1, rate=RATE,
                       input=True, frames_per_buffer=CHUNK)
    if config.MIC_INDEX is not None:
        open_kwargs["input_device_index"] = config.MIC_INDEX

    stream = pa.open(**open_kwargs)
    ep = EnergyEndpointer(RATE, CHUNK, config.VAD_RMS_THRESHOLD,
                          config.VAD_SILENCE_SECONDS, config.VAD_MIN_SECONDS,
                          config.VAD_NO_SPEECH_TIMEOUT, duration_s)
    frames = []
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            stop, _ = ep.feed(np.frombuffer(data, dtype=np.int16))
            if stop:
                break
    finally:
        sample_width = pa.get_sample_size(pyaudio.paInt16)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    with wave.open(save_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    return save_path


def wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# record command
# ---------------------------------------------------------------------------

def cmd_record(args):
    os.makedirs(AUDIO_DIR, exist_ok=True)

    if not os.path.exists(PHRASES_FILE):
        print(f"[error] Phrases file not found: {PHRASES_FILE}")
        sys.exit(1)

    with open(PHRASES_FILE, encoding="utf-8") as f:
        phrases = [l.strip() for l in f if l.strip()]

    references = {}
    if os.path.exists(REFERENCES_FILE):
        with open(REFERENCES_FILE, encoding="utf-8") as f:
            references = json.load(f)

    print(f"Recording {len(phrases)} phrases. Commands: Enter=accept, R=re-record, S=skip, Q=quit\n")
    print(f"Using mic index: {config.MIC_INDEX if config.MIC_INDEX is not None else 'default'}")
    print("Run 'python doctor.py --mic' to list devices and set JARVIS_MIC_INDEX.\n")

    for idx, phrase in enumerate(phrases):
        audio_id = f"{idx:03d}"
        wav_path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")

        if audio_id in references and os.path.exists(wav_path):
            ans = input(f"[{idx+1}/{len(phrases)}] Already recorded: '{phrase}'. Re-record? (y/N) ").strip().lower()
            if ans != "y":
                print("  Keeping existing recording.")
                continue

        while True:
            print(f"\n[{idx+1}/{len(phrases)}] Say: \"{phrase}\"")
            input("  Press Enter when ready, then speak...")
            print("  Recording...")
            try:
                record_phrase(wav_path)
                dur = wav_duration(wav_path)
                print(f"  Saved {dur:.1f}s → {wav_path}")
            except Exception as e:
                print(f"  [error] Recording failed: {e}")
                cont = input("  Retry? (Y/n) ").strip().lower()
                if cont == "n":
                    break
                continue

            cmd = input("  Accept (Enter), Re-record (R), Skip (S), Quit (Q): ").strip().upper()
            if cmd == "R":
                continue
            if cmd == "S":
                print("  Skipping.")
                break
            if cmd == "Q":
                _save_references(references)
                print("Saved references. Quitting.")
                return
            # Accept
            references[audio_id] = {"reference": phrase, "wav": wav_path}
            _save_references(references)
            print(f"  Accepted.")
            break

    _save_references(references)
    print(f"\nDone. {len(references)} recordings saved.")
    print(f"References: {REFERENCES_FILE}")
    print("Run: python bench/stt_bakeoff.py run")


def _save_references(refs: dict):
    with open(REFERENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(refs, f, indent=2)


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

def cmd_run(args):
    try:
        import jiwer
    except ImportError:
        print("[error] jiwer not installed. Run: pip install jiwer")
        sys.exit(1)

    if not os.path.exists(REFERENCES_FILE):
        print(f"[error] No references found at {REFERENCES_FILE}. Run 'record' first.")
        sys.exit(1)

    with open(REFERENCES_FILE, encoding="utf-8") as f:
        references = json.load(f)

    if not references:
        print("[error] references.json is empty. Run 'record' first.")
        sys.exit(1)

    # Collect available engines
    from audio.engines.registry import _get_registry
    registry = _get_registry()

    engine_filter = set(args.engines) if args.engines else None
    engines_to_run = []
    for name, engine in registry.items():
        if engine_filter and name not in engine_filter:
            continue
        if not engine.is_available():
            print(f"[skip] Engine '{name}' not available (missing package or API key).")
            continue
        engines_to_run.append(engine)

    if not engines_to_run:
        print("[error] No available engines to run.")
        sys.exit(1)

    print(f"Engines to run: {[e.name for e in engines_to_run]}")
    print(f"Recordings: {len(references)}")
    print(f"Repeats per recording: {args.repeats}\n")

    # Warm up
    print("Warming up engines...")
    warmup_wav = os.path.join(tempfile.gettempdir(), "jarvis_bakeoff_warmup.wav")
    _write_silence_wav(warmup_wav, 1.0)
    for engine in engines_to_run:
        try:
            engine.transcribe(warmup_wav)
            print(f"  [ok] {engine.name}")
        except Exception as e:
            print(f"  [warn] {engine.name} warmup failed: {e}")

    # Get system info
    import platform
    system_id = platform.node()
    device = config.WHISPER_DEVICE
    compute_type = config.WHISPER_COMPUTE_TYPE
    speaker_id = os.environ.get("JARVIS_SPEAKER_ID", "default")

    ts = time.strftime("%Y%m%dT%H%M%S")
    csv_path = os.path.join(BENCH_DIR, f"stt_results_{ts}.csv")

    fieldnames = [
        "timestamp", "system_id", "engine", "model", "device", "compute_type",
        "speaker_id", "audio_id", "audio_duration_s", "run_number",
        "latency_ms", "rtf", "wer", "cer", "exact_match",
        "reference_text", "transcribed_text",
    ]

    rows = []
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for engine in engines_to_run:
            model_name = (config.WHISPER_MODEL_SIZE
                          if engine.name == "faster_whisper" else engine.name)
            engine_rows = []

            for audio_id, ref_data in sorted(references.items()):
                wav_path = ref_data.get("wav", os.path.join(AUDIO_DIR, f"{audio_id}.wav"))
                reference = ref_data.get("reference", "")
                if not os.path.exists(wav_path):
                    print(f"  [skip] {wav_path} not found")
                    continue
                duration_s = wav_duration(wav_path)

                for run_num in range(1, args.repeats + 1):
                    try:
                        result = engine.transcribe(wav_path)
                        transcribed = result.text
                        latency_ms = result.latency_ms
                    except Exception as e:
                        print(f"  [error] {engine.name} / {audio_id} run {run_num}: {e}")
                        transcribed = ""
                        latency_ms = -1.0

                    ref_norm = normalize_text(reference)
                    hyp_norm = normalize_text(transcribed)
                    try:
                        wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm else 1.0
                        cer = jiwer.cer(ref_norm, hyp_norm) if ref_norm else 1.0
                    except Exception:
                        wer = 1.0
                        cer = 1.0
                    exact = int(ref_norm == hyp_norm)
                    rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 and latency_ms > 0 else -1.0

                    row = {
                        "timestamp": ts,
                        "system_id": system_id,
                        "engine": engine.name,
                        "model": model_name,
                        "device": device,
                        "compute_type": compute_type,
                        "speaker_id": speaker_id,
                        "audio_id": audio_id,
                        "audio_duration_s": round(duration_s, 3),
                        "run_number": run_num,
                        "latency_ms": round(latency_ms, 1),
                        "rtf": round(rtf, 4),
                        "wer": round(wer, 4),
                        "cer": round(cer, 4),
                        "exact_match": exact,
                        "reference_text": reference,
                        "transcribed_text": transcribed,
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    rows.append(row)
                    engine_rows.append(row)

            # Per-engine summary
            if engine_rows:
                latencies = [r["latency_ms"] for r in engine_rows if r["latency_ms"] > 0]
                wers = [r["wer"] for r in engine_rows]
                exact = [r["exact_match"] for r in engine_rows]
                import statistics
                med_lat = statistics.median(latencies) if latencies else float("nan")
                mean_wer = sum(wers) / len(wers)
                exact_pct = 100.0 * sum(exact) / len(exact)
                print(f"\n=== {engine.name} ({model_name}) ===")
                print(f"  Median latency : {med_lat:.0f} ms")
                print(f"  Mean WER       : {mean_wer:.3f} ({mean_wer*100:.1f}%)")
                print(f"  Exact match    : {exact_pct:.1f}%  ({sum(exact)}/{len(exact)})")

    print(f"\nResults saved to: {csv_path}")
    print("Normalization: lowercase, strip punctuation, collapse whitespace. Numerals kept as-is.")


def _write_silence_wav(path: str, duration_s: float):
    samples = np.zeros(int(RATE * duration_s), dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="STT bake-off tool")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("record", help="Record phrases from microphone")

    run_p = sub.add_parser("run", help="Run engines over recordings")
    run_p.add_argument("--engines", nargs="*", help="Engine names to run (default: all available)")
    run_p.add_argument("--repeats", type=int, default=1, help="Number of runs per recording (default 1)")

    args = ap.parse_args()
    if args.command == "record":
        cmd_record(args)
    elif args.command == "run":
        cmd_run(args)
