"""
Unit tests for bench/record.py audio capture, filename generation, and logging.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.record import (
    LOG_COLUMNS,
    analyze_audio,
    get_audio_filename,
    get_audio_path,
    load_commands_for_speaker,
    log_recording,
    run_recording_session,
    save_wav,
)


class TestBenchRecord(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_filename_generation(self):
        filename = get_audio_filename("t001", "S1")
        self.assertEqual(filename, "t001_S1.wav")

        path = get_audio_path(self.out_dir, "t002", "S2")
        expected = os.path.join(self.out_dir, "t002_S2.wav")
        self.assertEqual(path, expected)

    def test_audio_analysis_normal(self):
        # 16000 samples of 10000 peak (1.0 second)
        samples = (np.ones(16000, dtype=np.int16) * 10000).tobytes()
        res = analyze_audio(samples, rate=16000)
        self.assertEqual(res["duration_s"], 1.0)
        self.assertEqual(res["peak_level"], 10000)
        self.assertFalse(res["clipped"])
        self.assertFalse(res["is_low"])

    def test_audio_analysis_clipped(self):
        # Samples hitting full scale 32767
        arr = np.zeros(16000, dtype=np.int16)
        arr[100] = 32767
        res = analyze_audio(arr.tobytes(), rate=16000)
        self.assertTrue(res["clipped"])
        self.assertEqual(res["peak_level"], 32767)

    def test_audio_analysis_low_level(self):
        # Very quiet audio (peak 250 < 1000)
        arr = (np.ones(16000, dtype=np.int16) * 250).tobytes()
        res = analyze_audio(arr, rate=16000)
        self.assertFalse(res["clipped"])
        self.assertTrue(res["is_low"])

    def test_log_recording_csv(self):
        log_csv = os.path.join(self.out_dir, "recordings.csv")
        log_recording(
            log_csv,
            cmd_id="t001",
            speaker="S1",
            audio_path="bench/audio/t001_S1.wav",
            duration_s=2.34,
            peak_level=15200,
            clipped=False,
            timestamp="2026-10-21T10:00:00",
        )

        self.assertTrue(os.path.exists(log_csv))
        with open(log_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, LOG_COLUMNS)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["id"], "t001")
            self.assertEqual(row["speaker"], "S1")
            self.assertEqual(row["path"], "bench/audio/t001_S1.wav")
            self.assertEqual(row["duration_s"], "2.34")
            self.assertEqual(row["peak_level"], "15200")
            self.assertEqual(row["clipped"], "False")
            self.assertEqual(row["timestamp"], "2026-10-21T10:00:00")

    def test_load_commands_for_speaker_and_noisy_flag(self):
        csv_path = os.path.join(ROOT, "jarvis_test_set_template.csv")
        if not os.path.exists(csv_path):
            self.skipTest("jarvis_test_set_template.csv not present")

        # Standard load for speaker S1
        commands = load_commands_for_speaker(csv_path, speaker="S1", noisy=False)
        self.assertGreater(len(commands), 0)
        for c in commands:
            self.assertIn("id", c)
            self.assertIn("text", c)

        # Noisy load limits to first 30 IDs
        noisy_commands = load_commands_for_speaker(csv_path, speaker="S1", noisy=True)
        self.assertLessEqual(len(noisy_commands), 30)

    def test_run_recording_session_auto_mode(self):
        csv_path = os.path.join(ROOT, "jarvis_test_set_template.csv")
        if not os.path.exists(csv_path):
            self.skipTest("jarvis_test_set_template.csv not present")

        # Run automated test session for speaker S1 on noisy set (subset of commands)
        run_recording_session(
            speaker="S1",
            csv_path=csv_path,
            out_dir=self.out_dir,
            noisy=True,
            auto_test_mode=True,
        )

        wav_path = os.path.join(self.out_dir, "t001_S1.wav")
        self.assertTrue(os.path.exists(wav_path))

        log_csv = os.path.join(self.out_dir, "recordings.csv")
        self.assertTrue(os.path.exists(log_csv))


if __name__ == "__main__":
    unittest.main()
