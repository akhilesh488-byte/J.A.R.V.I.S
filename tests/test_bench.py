"""
Unit tests for the J.A.R.V.I.S. Benchmark Suite:
- Scoring logic (tool_ok, args_ok, valid_json, false_tool_call, unsupported, confirmation)
- CSV schema and dataset validation
"""

from __future__ import annotations

import csv
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.dataset import (
    DatasetRow,
    load_dataset,
    load_dev_set,
    load_test_set,
    load_dev60_labels,
)
from bench.env_capture import capture_environment
from bench.mock_tools import BenchmarkMockTools
from bench.run import CSV_COLUMNS, run_benchmark
from bench.scoring import (
    check_unsupported_reply,
    check_whatsapp_confirmation,
    rescore_results_csv,
    score_call,
)


class TestBenchScoring(unittest.TestCase):
    def test_tool_and_args_exact_match(self):
        calls = [{"tool": "open_app", "args": {"app_name": "Spotify"}}]
        score = score_call(
            expected_tool="open_app",
            expected_args={"app_name": "spotify"},
            calls=calls,
            valid_json=True,
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])
        self.assertTrue(score["valid_json"])
        self.assertFalse(score["false_tool_call"])

    def test_tool_args_mismatch(self):
        calls = [{"tool": "open_app", "args": {"app_name": "chrome"}}]
        score = score_call(
            expected_tool="open_app",
            expected_args={"app_name": "spotify"},
            calls=calls,
            valid_json=True,
        )
        self.assertTrue(score["tool_ok"])
        self.assertFalse(score["args_ok"])
        self.assertFalse(score["false_tool_call"])

    def test_extra_model_args_ignored(self):
        calls = [{"tool": "set_timer", "args": {"seconds": 300, "label": "Tea", "extra_param": 1}}]
        score = score_call(
            expected_tool="set_timer",
            expected_args={"seconds": 300},
            calls=calls,
            valid_json=True,
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])

    def test_false_tool_call_when_expected_none(self):
        calls = [{"tool": "open_app", "args": {"app_name": "question"}}]
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=calls,
            valid_json=True,
            category="adversarial",
        )
        self.assertFalse(score["tool_ok"])
        self.assertFalse(score["args_ok"])
        self.assertTrue(score["false_tool_call"])

    def test_expected_none_no_tool_called(self):
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=[],
            valid_json=False,
            reply="The capital of France is Paris.",
            category="reasoning",
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])
        self.assertFalse(score["false_tool_call"])

    def test_unsupported_request_claim_success_fails(self):
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=[],
            valid_json=False,
            reply="I have made the pizza and baked it for you!",
            category="unsupported",
        )
        self.assertFalse(score["tool_ok"])
        self.assertFalse(score["unsupported_ok"])

    def test_unsupported_request_polite_refusal_passes(self):
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=[],
            valid_json=False,
            reply="I'm afraid I cannot make a pizza, sir. I can only assist with desktop tasks.",
            category="unsupported",
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["unsupported_ok"])

    def test_whatsapp_confirmation_rule_failure(self):
        calls = [{"tool": "send_whatsapp_message", "args": {"recipient": "Bob", "message": "hello"}}]
        # History without confirmation question
        history = [{"role": "user", "content": "send a message"}]
        score = score_call(
            expected_tool="send_whatsapp_message",
            expected_args={"recipient": "Bob", "message": "hello"},
            calls=calls,
            history=history,
        )
        self.assertFalse(score["confirmation_ok"])
        self.assertFalse(score["tool_ok"])

    def test_whatsapp_confirmation_rule_success(self):
        calls = [{"tool": "send_whatsapp_message", "args": {"recipient": "Bob", "message": "hello"}}]
        history = [
            {"role": "user", "content": "send whatsapp to Bob saying hello"},
            {"role": "assistant", "content": "Shall I send the WhatsApp message to Bob?"},
        ]
        score = score_call(
            expected_tool="send_whatsapp_message",
            expected_args={"recipient": "Bob", "message": "hello"},
            calls=calls,
            history=history,
        )
        self.assertTrue(score["confirmation_ok"])
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])

    def test_wildcard_placeholder_matching(self):
        calls = [{"tool": "create_event", "args": {
            "summary": "project presentation",
            "start_time": "2026-10-24T09:00:00",
            "end_time": "2026-10-24T10:00:00",
        }}]
        score = score_call(
            expected_tool="create_event",
            expected_args={
                "summary": "project presentation",
                "start_time": "<tomorrow 09:00 ISO from the fixed clock>",
                "end_time": "<+1h>",
            },
            calls=calls,
            valid_json=True,
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])

    def test_contact_recipient_alias(self):
        calls = [{"tool": "send_whatsapp_message", "args": {"recipient": "Alice", "message": "Hi"}}]
        history = [
            {"role": "user", "content": "send whatsapp"},
            {"role": "assistant", "content": "Shall I send the WhatsApp message to Alice?"},
        ]
        score = score_call(
            expected_tool="send_whatsapp_message",
            expected_args={"contact": "Alice", "message": "Hi"},
            calls=calls,
            valid_json=True,
            history=history,
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])
        self.assertTrue(score["confirmation_ok"])


class TestBenchCsvAndDataset(unittest.TestCase):
    def test_dataset_loader(self):
        path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
        rows = load_dataset(path)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0].id, "d01")
        self.assertEqual(rows[0].expected_tool, "open_app")
        self.assertIsInstance(rows[0].expected_args, dict)

    def test_template_csv_loader(self):
        csv_path = os.path.join(ROOT, "jarvis_test_set_template.csv")
        rows = load_dataset(csv_path)
        self.assertEqual(len(rows), 120)
        self.assertEqual(rows[0].id, "t001")
        # t001 is injected with ground truth from dev60_proposed_labels.csv: manage_browser
        self.assertEqual(rows[0].expected_tool, "manage_browser")
        self.assertIsInstance(rows[0].expected_args, dict)
        self.assertEqual(rows[0].expected_args.get("action"), "open_url")

    def test_mock_tools(self):
        mock = BenchmarkMockTools()
        table = mock.table()
        self.assertIn("open_app", table)
        self.assertIn("add_todo", table)
        self.assertIn("delegate_to_researcher", table)
        res = table["open_app"](app_name="vscode")
        # Should return realistic string from tool_results.json: "Opening vscode."
        self.assertEqual(res, "Opening vscode.")
        self.assertEqual(len(mock.calls), 1)
        self.assertEqual(mock.calls[0]["tool"], "open_app")
        self.assertEqual(mock.calls[0]["args"], {"app_name": "vscode"})

    def test_csv_schema_and_runner(self):
        dataset_path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
        out_dir = os.path.join(ROOT, "results", "test_out")
        os.makedirs(out_dir, exist_ok=True)
        csv_target = os.path.join(out_dir, "unit_test_exp_stub_TestBox.csv")
        if os.path.exists(csv_target):
            os.remove(csv_target)

        csv_path = run_benchmark(
            experiment="unit_test_exp",
            machine="TestBox",
            provider_name="stub",
            out_dir=out_dir,
            dataset_path=dataset_path,
            repeats=1,
            warmup=0,
        )

        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, CSV_COLUMNS)
            rows = list(reader)
            self.assertEqual(len(rows), 10)
            for r in rows:
                self.assertIn(r["tool_ok"], ("True", "False"))
                self.assertIn(r["args_ok"], ("True", "False"))
                self.assertEqual(r["experiment"], "unit_test_exp")
                self.assertEqual(r["machine"], "TestBox")

    def test_environment_capture(self):
        dataset_path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
        out_json = os.path.join(ROOT, "results", "test_out", "env.json")
        env = capture_environment(dataset_path=dataset_path, output_path=out_json)

        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIn("git_commit", env)
        self.assertIn("hashes", env)
        self.assertIsNotNone(env["hashes"]["tool_registry_sha256"])
        self.assertIsNotNone(env["hashes"]["dataset_sha256"])
        self.assertTrue(os.path.exists(out_json))

    def test_benchmark_with_audio_dir(self):
        # Create a mini dataset with t001
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = os.path.join(tmpdir, "mini.jsonl")
            with open(dataset_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": "t001",
                    "category": "tool",
                    "split": "test",
                    "text": "open spotify",
                    "expected_tool": "open_app",
                    "expected_args": {"app_name": "spotify"},
                }) + "\n")

            out_dir = os.path.join(tmpdir, "out")
            audio_dir = os.path.join(ROOT, "bench", "audio")

            csv_path = run_benchmark(
                experiment="audio_test_exp",
                machine="TestBox",
                provider_name="stub",
                out_dir=out_dir,
                dataset_path=dataset_file,
                repeats=1,
                warmup=0,
                audio_dir=audio_dir,
            )

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["command_id"], "t001")
                self.assertTrue(rows[0]["audio_path"].endswith(".wav"))
                self.assertGreater(float(rows[0]["stt_ms"]), 0.0)
                self.assertTrue(len(rows[0]["text"]) > 0)

    def test_dev_and_test_isolation(self):
        dev_rows = load_dev_set()
        test_rows = load_test_set()
        self.assertEqual(len(dev_rows), 60)
        self.assertEqual(len(test_rows), 60)

        dev_ids = {r.id for r in dev_rows}
        test_ids = {r.id for r in test_rows}

        # Strict separation: zero overlap
        self.assertEqual(dev_ids & test_ids, set())
        # All 120 accounted for
        self.assertEqual(len(dev_ids | test_ids), 120)

        # All dev rows must have split == 'dev'
        for r in dev_rows:
            self.assertEqual(r.split, "dev")
        # All test rows must have split == 'test'
        for r in test_rows:
            self.assertEqual(r.split, "test")

        # Ground truth injection checks on dev set
        dev_map = {r.id: r for r in dev_rows}
        self.assertEqual(dev_map["t001"].expected_tool, "manage_browser")
        self.assertEqual(dev_map["t001"].expected_args, {"action": "open_url", "query": "youtube.com"})
        self.assertEqual(dev_map["t004"].expected_tool, "add_todo")
        self.assertEqual(dev_map["t004"].expected_args, {"task": "going to the gym at 7pm"})
        self.assertIsNone(dev_map["t012"].expected_tool)
        self.assertEqual(dev_map["t013"].expected_tool, "set_timer")
        self.assertEqual(dev_map["t013"].expected_args, {"seconds": 1200})
        self.assertEqual(dev_map["t016"].expected_tool, "set_alarm_at")
        self.assertEqual(dev_map["t016"].expected_args, {"hour": 7, "minute": 30})
        self.assertIsNone(dev_map["t024"].expected_tool)

    def test_split_runner_independent_execution(self):
        out_dir = os.path.join(ROOT, "results", "test_out")
        dev_csv = run_benchmark(
            experiment="split_test",
            machine="BoxA",
            provider_name="stub",
            out_dir=out_dir,
            split="dev",
            repeats=1,
            warmup=0,
        )
        self.assertTrue(os.path.exists(dev_csv))
        self.assertIn("_dev_", dev_csv)
        with open(dev_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dev_rows = list(reader)
        self.assertEqual(len(dev_rows), 60)
        for r in dev_rows:
            self.assertEqual(r["split"], "dev")

        test_csv = run_benchmark(
            experiment="split_test",
            machine="BoxA",
            provider_name="stub",
            out_dir=out_dir,
            split="test",
            repeats=1,
            warmup=0,
        )
        self.assertTrue(os.path.exists(test_csv))
        self.assertIn("_test_", test_csv)
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            test_rows = list(reader)
        self.assertEqual(len(test_rows), 60)
        for r in test_rows:
            self.assertEqual(r["split"], "test")

        # Dev and test execution results do not contaminate each other
        dev_cids = {r["command_id"] for r in dev_rows}
        test_cids = {r["command_id"] for r in test_rows}
        self.assertEqual(dev_cids & test_cids, set())

    def test_rescore_results_csv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_csv = os.path.join(tmpdir, "sample.csv")
            with open(sample_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerow({
                    "command_id": "t001",
                    "category": "tool",
                    "split": "test",
                    "tool_called": "manage_browser",
                    "args_called": json.dumps({"action": "open_url", "query": "youtube.com"}),
                    "valid_json": "True",
                    "reply": "Opening youtube.",
                    "tool_ok": "False",
                    "args_ok": "False",
                })
            rescored = rescore_results_csv(sample_csv)
            self.assertEqual(len(rescored), 1)
            row = rescored[0]
            self.assertEqual(row["split"], "dev")
            self.assertEqual(row["expected_tool"], "manage_browser")
            self.assertEqual(row["tool_ok"], "True")
            self.assertEqual(row["args_ok"], "True")


if __name__ == "__main__":
    unittest.main()

