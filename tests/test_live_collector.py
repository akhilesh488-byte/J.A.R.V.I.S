"""
Unit tests for bench/live_collector.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.live_collector import (
    CSV_HEADERS,
    append_live_record,
    ensure_csv_initialized,
    process_audio_interaction,
    run_live_collector,
)


class TestLiveCollector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "live_dataset.csv")
        self.audio_dir = os.path.join(self.temp_dir.name, "audio")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ensure_csv_initialized(self):
        ensure_csv_initialized(self.csv_path)
        self.assertTrue(os.path.exists(self.csv_path))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, CSV_HEADERS)

    def test_append_live_record(self):
        record = {
            "id": "live_abc123",
            "category": "live_collection",
            "history": "[]",
            "text": "open spotify",
            "expected_tool": "open_app",
            "expected_args": '{"app_name": "spotify"}',
            "record_by": "S1",
            "audio_path": "bench/audio/live_cmd_abc123.wav",
        }
        append_live_record(self.csv_path, record)

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, CSV_HEADERS)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "live_abc123")
            self.assertEqual(rows[0]["category"], "live_collection")
            self.assertEqual(rows[0]["text"], "open spotify")
            self.assertEqual(rows[0]["expected_tool"], "open_app")
            self.assertEqual(rows[0]["record_by"], "S1")

    @patch("audio.stt.transcribe")
    @patch("bench.live_collector.route_command")
    def test_process_audio_interaction(self, mock_route, mock_transcribe):
        mock_transcribe.return_value = "set a timer for ten minutes"
        mock_route_res = MagicMock()
        mock_route_res.tool_called = "set_timer"
        mock_route_res.args = {"seconds": 600}
        mock_route_res.reply = "Timer set for 10 minutes, sir."
        mock_route.return_value = mock_route_res

        dummy_pcm = (np.ones(16000, dtype=np.int16) * 1000).tobytes()
        rec = process_audio_interaction(
            pcm_bytes=dummy_pcm,
            speaker="S2",
            audio_dir=self.audio_dir,
            csv_path=self.csv_path,
            cmd_uuid="test1234",
        )

        self.assertIsNotNone(rec)
        self.assertEqual(rec["id"], "live_test1234")
        self.assertEqual(rec["expected_tool"], "set_timer")
        self.assertEqual(json.loads(rec["expected_args"]), {"seconds": 600})
        self.assertEqual(rec["record_by"], "S2")

        wav_path = os.path.join(self.audio_dir, "live_cmd_test1234.wav")
        self.assertTrue(os.path.exists(wav_path))


if __name__ == "__main__":
    unittest.main()
