"""
Unit tests for the Faster-Whisper STT pipeline in providers.py:
1. Hardware Explicit Typing (cuda float16 with graceful cpu int8 fallback)
2. Context Seeding (domain vocabulary priming)
3. Algorithmic Repetition Guards (condition_on_previous_text=False, compression_ratio_threshold=2.4, logprob_threshold=-1.0)
4. Post-Processing N-Gram Filter (deterministic bigram loop pruning)
5. Execution Timeout (strict 10-second timeout protection)
"""

from __future__ import annotations

import collections
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.providers import (
    DEFAULT_WHISPER_INITIAL_PROMPT,
    FasterWhisperPipeline,
    FasterWhisperSTTProvider,
    filter_repetitive_ngrams,
    get_whisper_pipeline,
)


class DummySegment:
    def __init__(self, text: str):
        self.text = text


class TestHardwareExplicitTyping(unittest.TestCase):
    @patch("faster_whisper.WhisperModel")
    def test_cuda_success(self, mock_model_cls):
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance

        pipeline = FasterWhisperPipeline(model_size="base")
        self.assertEqual(pipeline.device, "cuda")
        self.assertEqual(pipeline.compute_type, "float16")
        mock_model_cls.assert_called_with("base", device="cuda", compute_type="float16", cpu_threads=4, num_workers=1)

    @patch("faster_whisper.WhisperModel")
    def test_cuda_oom_or_failure_fallback_to_cpu_int8(self, mock_model_cls):
        # First call (CUDA) fails with RuntimeError / OOM; second call (CPU) succeeds
        mock_cuda_fail = RuntimeError("CUDA out of memory or device not available")
        mock_cpu_instance = MagicMock()
        mock_model_cls.side_effect = [mock_cuda_fail, mock_cpu_instance]

        pipeline = FasterWhisperPipeline(model_size="base")
        self.assertEqual(pipeline.device, "cpu")
        self.assertEqual(pipeline.compute_type, "int8")
        self.assertEqual(mock_model_cls.call_count, 2)
        # Verify the two calls
        calls = mock_model_cls.call_args_list
        self.assertEqual(calls[0].kwargs["device"], "cuda")
        self.assertEqual(calls[0].kwargs["compute_type"], "float16")
        self.assertEqual(calls[1].kwargs["device"], "cpu")
        self.assertEqual(calls[1].kwargs["compute_type"], "int8")


class TestContextSeeding(unittest.TestCase):
    def test_default_prompt_contains_domain_terms(self):
        expected_terms = [
            "gym", "to-dos", "PyTorch", "LangChain", "GitHub", "WhatsApp", "VS Code", "J.A.R.V.I.S."
        ]
        for term in expected_terms:
            self.assertIn(term, DEFAULT_WHISPER_INITIAL_PROMPT)

    def test_transcribe_passes_initial_prompt(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([DummySegment("test command")], {})
        pipeline = FasterWhisperPipeline(model=mock_model)

        pipeline.transcribe("fake.wav")
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args.kwargs
        self.assertEqual(call_kwargs["initial_prompt"], DEFAULT_WHISPER_INITIAL_PROMPT)


class TestAlgorithmicRepetitionGuards(unittest.TestCase):
    def test_repetition_guards_passed_to_transcribe(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([DummySegment("test command")], {})
        pipeline = FasterWhisperPipeline(model=mock_model)

        pipeline.transcribe("fake.wav")
        call_kwargs = mock_model.transcribe.call_args.kwargs

        self.assertFalse(call_kwargs["condition_on_previous_text"])
        self.assertEqual(call_kwargs["compression_ratio_threshold"], 2.4)
        # Check either logprob_threshold or log_prob_threshold is set to -1.0
        lp_val = call_kwargs.get("logprob_threshold", call_kwargs.get("log_prob_threshold"))
        self.assertEqual(lp_val, -1.0)


class TestPostProcessingNGramFilter(unittest.TestCase):
    def test_clean_text_unchanged(self):
        text = "open spotify and play my favorite songs"
        self.assertEqual(filter_repetitive_ngrams(text), text)

    def test_empty_or_short_text(self):
        self.assertEqual(filter_repetitive_ngrams(""), "")
        self.assertEqual(filter_repetitive_ngrams("hello"), "hello")

    def test_repetitive_bigram_pruned(self):
        # 6 consecutive repetitions of 'to-dos of'
        raw = "i want you to list my to-dos of to-dos of to-dos of to-dos of to-dos of to-dos of"
        pruned = filter_repetitive_ngrams(raw, max_consecutive=4)
        # Should retain only the first 4 repetitions and prune the 5th and 6th
        expected = "i want you to list my to-dos of to-dos of to-dos of to-dos of"
        self.assertEqual(pruned, expected)

    def test_unigram_loop_pruned(self):
        # 6 consecutive 'knock' words form 5 consecutive ('knock', 'knock') bigrams
        raw = "knock knock knock knock knock knock"
        pruned = filter_repetitive_ngrams(raw, max_consecutive=4)
        expected = "knock knock knock knock"
        self.assertEqual(pruned, expected)

    def test_four_repetitions_kept(self):
        # Exactly 4 repetitions of 'to-dos of' should be preserved
        raw = "my to-dos of to-dos of to-dos of to-dos of"
        self.assertEqual(filter_repetitive_ngrams(raw, max_consecutive=4), raw)


class TestExecutionTimeout(unittest.TestCase):
    def test_successful_execution_within_timeout(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([DummySegment("turn on lights")], {})
        pipeline = FasterWhisperPipeline(model=mock_model)

        result = pipeline.transcribe("fake.wav", timeout=2.0)
        self.assertEqual(result, "turn on lights")

    def test_hanging_execution_triggers_timeout(self):
        mock_model = MagicMock()

        def slow_transcribe(*args, **kwargs):
            time.sleep(1.0)
            return ([DummySegment("too late")], {})

        mock_model.transcribe.side_effect = slow_transcribe
        pipeline = FasterWhisperPipeline(model=mock_model)

        with self.assertRaises(TimeoutError):
            pipeline.transcribe("fake.wav", timeout=0.1)


if __name__ == "__main__":
    unittest.main()
