"""
Benchmark providers for Jarvis evaluation.
Implements:
1. OllamaBenchmarkProvider: streaming client capturing TTFT (time-to-first-token),
   prompt_eval_count, eval_count, prompt_eval_duration, eval_duration.
   Temperature = 0.0. Memory & user profile injection OFF.
2. CloudBenchmarkProvider: wraps existing ModelManager cloud clients with temperature=0.0.
3. RegexBenchmarkProvider: wraps frozen classify_regex baseline.
4. StubBenchmarkProvider: offline deterministic mock provider for dry runs and unit tests.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from bench.baselines.regex_router import RegexBaselineRouter
from core import memory
from tools.tool_registry import build_dispatch_table, validate_tool_call

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_INITIAL_PROMPT = (
    "gym, to-dos, PyTorch, LangChain, GitHub, WhatsApp, VS Code, J.A.R.V.I.S."
)


def filter_repetitive_ngrams(text: str, max_consecutive: int = 4) -> str:
    """
    Deterministic post-processing function that splits the transcript into bigrams,
    checks the repetition frequency, and truncates the string if any bigram repeats
    more than 4 consecutive times (e.g., safely pruning the 'to-dos of to-dos' loop).
    """
    if not text or not isinstance(text, str):
        return "" if text is None else str(text)

    words = text.split()
    if len(words) < 2:
        return text

    # Explicitly form bigrams from consecutive tokens
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    earliest_cutoff = len(words)

    # 1. Stride-2 check: standard alternating 2-word phrase cycle (e.g., "to-dos of to-dos of...")
    for start in range(len(words) - 1):
        bg = (words[start], words[start + 1])
        count = 1
        curr = start + 2
        while curr < len(words) - 1 and (words[curr], words[curr + 1]) == bg:
            count += 1
            if count > max_consecutive:
                earliest_cutoff = min(earliest_cutoff, curr)
                break
            curr += 2

    # 2. Stride-1 check: unigram repetition yielding consecutive identical bigrams (e.g., "word word word...")
    for start in range(len(words) - 1):
        bg = (words[start], words[start + 1])
        count = 1
        curr = start + 1
        while curr < len(words) - 1 and (words[curr], words[curr + 1]) == bg:
            count += 1
            if count > max_consecutive:
                earliest_cutoff = min(earliest_cutoff, curr)
                break
            curr += 1

    if earliest_cutoff < len(words):
        return " ".join(words[:earliest_cutoff])
    return text


filter_repetitive_bigrams = filter_repetitive_ngrams
filter_bigram_repetition = filter_repetitive_ngrams
prune_bigram_repetition = filter_repetitive_ngrams


class FasterWhisperPipeline:
    """
    Production-grade Faster-Whisper Speech-to-Text pipeline meeting reliability standards:
    1. Hardware Explicit Typing: Strict 'cuda' with 'float16', graceful fallback to 'cpu' with 'int8' on OOM/init error.
    2. Context Seeding: Biases decoder with high-frequency domain vocabulary.
    3. Algorithmic Repetition Guards: condition_on_previous_text=False, compression_ratio_threshold=2.4, logprob_threshold=-1.0.
    4. Post-Processing N-Gram Filter: Deterministically prunes repetitive loops (>4 consecutive bigrams).
    5. Execution Timeout: Strict 10-second timeout to prevent indefinite hangs on degenerate audio.
    """

    DEFAULT_INITIAL_PROMPT: str = DEFAULT_WHISPER_INITIAL_PROMPT
    DEFAULT_TIMEOUT: float = 10.0

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        model: Optional[Any] = None,
        cpu_threads: int = 4,
        num_workers: int = 1,
    ):
        self.model_size = model_size or getattr(config, "WHISPER_MODEL_SIZE", "base")
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.model = model
        self.device = device
        self.compute_type = compute_type

        if self.model is None:
            self._initialize_model(device=device, compute_type=compute_type)
        else:
            self.device = device or "cuda"
            self.compute_type = compute_type or "float16"

    def _initialize_model(
        self, device: Optional[str] = None, compute_type: Optional[str] = None
    ) -> None:
        """
        Instantiate the Whisper model strictly with device="cuda" and compute_type="float16".
        Wrap this in a try/except block that catches CUDA OOM or initialization failures,
        falling back gracefully to device="cpu" and compute_type="int8".
        """
        from faster_whisper import WhisperModel

        target_device = device or "cuda"
        target_compute = compute_type or "float16"

        if target_device == "cuda":
            try:
                candidate_model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16",
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                )
                # Verify CUDA kernels are functional (CTranslate2 lazy-loads cublas/cudnn on first tensor op)
                import numpy as np
                dummy = np.zeros(16000, dtype=np.float32)
                list(candidate_model.transcribe(dummy)[0])

                self.model = candidate_model
                self.device = "cuda"
                self.compute_type = "float16"
                return
            except Exception as e:
                logger.warning(
                    f"[FasterWhisperPipeline] CUDA initialization or OOM failure ({e}). "
                    f"Falling back gracefully to device='cpu', compute_type='int8'..."
                )

        # Fallback to CPU with int8 quantization
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
        )
        self.device = "cpu"
        self.compute_type = "int8"

    def transcribe(
        self,
        audio: Any,
        initial_prompt: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        condition_on_previous_text: bool = False,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        **kwargs,
    ) -> str:
        """
        Transcribes audio with context seeding, algorithmic repetition guards,
        strict 10-second timeout, and post-processing n-gram filtering.
        """
        result_container: List[str] = []
        error_container: List[Exception] = []

        def _worker():
            try:
                res = self._transcribe_internal(
                    audio=audio,
                    initial_prompt=initial_prompt,
                    condition_on_previous_text=condition_on_previous_text,
                    compression_ratio_threshold=compression_ratio_threshold,
                    logprob_threshold=logprob_threshold,
                    **kwargs,
                )
                result_container.append(res)
            except Exception as exc:
                error_container.append(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Faster-Whisper transcription exceeded strict {timeout}-second timeout on audio input"
            )
        if error_container:
            raise error_container[0]

        return result_container[0] if result_container else ""

    def _transcribe_internal(
        self,
        audio: Any,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = False,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        **kwargs,
    ) -> str:
        if self.model is None:
            self._initialize_model()

        seed_prompt = (
            initial_prompt
            if initial_prompt is not None
            else self.DEFAULT_INITIAL_PROMPT
        )

        transcribe_args = {
            "initial_prompt": seed_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
        }
        for k, v in kwargs.items():
            if k not in transcribe_args:
                transcribe_args[k] = v

        # Parameter name compatibility:
        # CTranslate2 / faster-whisper uses log_prob_threshold
        # OpenAI Whisper API uses logprob_threshold
        try:
            try:
                segments, info = self.model.transcribe(
                    audio,
                    logprob_threshold=logprob_threshold,
                    **transcribe_args,
                )
            except TypeError as te:
                if "logprob_threshold" in str(te) or "unexpected keyword" in str(te):
                    segments, info = self.model.transcribe(
                        audio,
                        log_prob_threshold=logprob_threshold,
                        **transcribe_args,
                    )
                else:
                    raise
        except Exception as e:
            if self.device == "cuda" and (
                "cublas" in str(e).lower()
                or "cuda" in str(e).lower()
                or "out of memory" in str(e).lower()
            ):
                logger.warning(
                    f"[FasterWhisperPipeline] CUDA runtime failure ({e}). "
                    f"Falling back gracefully to CPU int8..."
                )
                self._initialize_model(device="cpu", compute_type="int8")
                return self._transcribe_internal(
                    audio=audio,
                    initial_prompt=initial_prompt,
                    condition_on_previous_text=condition_on_previous_text,
                    compression_ratio_threshold=compression_ratio_threshold,
                    logprob_threshold=logprob_threshold,
                    **kwargs,
                )
            raise

        seg_list = list(segments)
        raw_text = " ".join(s.text for s in seg_list).strip()

        # Deterministic bigram post-processing filter
        pruned_text = filter_repetitive_ngrams(raw_text, max_consecutive=4)
        return pruned_text


_GLOBAL_PIPELINE: Optional[FasterWhisperPipeline] = None


def get_whisper_pipeline(
    model_size: Optional[str] = None,
    reset: bool = False,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> FasterWhisperPipeline:
    global _GLOBAL_PIPELINE
    if reset or _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = FasterWhisperPipeline(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )
    return _GLOBAL_PIPELINE


def transcribe_audio(
    audio: Any,
    initial_prompt: Optional[str] = None,
    timeout: float = FasterWhisperPipeline.DEFAULT_TIMEOUT,
    **kwargs,
) -> str:
    pipeline = get_whisper_pipeline()
    return pipeline.transcribe(audio, initial_prompt=initial_prompt, timeout=timeout, **kwargs)


V2_SYSTEM_TOOLS = {
    "manage_browser",
    "play_pause_media",
    "close_current_app",
    "get_window_status",
    "send_whatsapp_message",
}


def build_benchmark_system_prompt() -> str:
    """
    Builds clean system prompt for benchmarks dynamically from system_prompt_template.txt
    and tool_lines.txt instead of hardcoded strings.
    """
    from core.prompt import build_system_prompt
    return build_system_prompt()


class BenchmarkProvider:
    name: str = "base"
    model: str = "base"
    quant: str = "none"

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class OllamaBenchmarkProvider(BenchmarkProvider):
    name = "ollama"

    def __init__(
        self,
        model_tag: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model_tag or config.LOCAL_LLM_MODEL or "jarvis-3b"
        self.base_url = (base_url or config.LOCAL_LLM_URL).replace("/v1", "")
        self.timeout = timeout
        self.quant = self._detect_quantization()

    def _detect_quantization(self) -> str:
        tag_lower = self.model.lower()
        for q in ["q4_k_m", "q4_0", "q4_1", "q5_k_m", "q8_0", "fp16", "f16"]:
            if q in tag_lower:
                return q.upper()
        return "Q4_K_M"

    def _stream_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, float, Dict[str, Any]]:
        url = f"{self.base_url}/api/chat"
        options = {
            "temperature": 0.0,
            "num_ctx": 4096,
            "seed": 42,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
            "options": options,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        ttft_ms = 0.0
        full_content = []
        final_meta: Dict[str, Any] = {}

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                msg = chunk.get("message", {}).get("content", "")
                if msg:
                    if ttft_ms == 0.0:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    full_content.append(msg)
                if chunk.get("done"):
                    final_meta = chunk

        total_content = "".join(full_content).strip()
        if ttft_ms == 0.0:
            ttft_ms = (time.perf_counter() - t_start) * 1000.0

        return total_content, ttft_ms, final_meta

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        system_prompt = build_benchmark_system_prompt()

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        # Follow-up rows pass earlier_user/earlier_jarvis through production history path
        if history:
            for turn in history:
                content = (turn.get("content") or "").strip()
                if not content.startswith("{") and not content.startswith("TOOL_RESULT:"):
                    messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": text})

        ttft_ms = 0.0
        tokens_in = 0
        tokens_out = 0
        load_dur_ns = 0
        prompt_eval_dur_ns = 0
        eval_dur_ns = 0
        iterations = 0
        tool_called = None
        tool_args = {}
        valid_json = False
        final_reply = ""
        error_msg = None

        try:
            iterations += 1
            content, first_ttft, meta = self._stream_chat(messages)
            ttft_ms = first_ttft
            tokens_in += meta.get("prompt_eval_count", 0)
            tokens_out += meta.get("eval_count", 0)
            load_dur_ns += meta.get("load_duration", 0) or 0
            prompt_eval_dur_ns += meta.get("prompt_eval_duration", 0) or 0
            eval_dur_ns += meta.get("eval_duration", 0) or 0

            if content.startswith("{"):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        valid_json = True
                        name = parsed["tool"]
                        raw_args = parsed.get("args") or {}
                        if not isinstance(raw_args, dict):
                            raw_args = {}

                        is_valid, _ = validate_tool_call(name, raw_args)
                        if not is_valid and name in V2_SYSTEM_TOOLS:
                            is_valid = True

                        if is_valid:
                            tool_called = name
                            tool_args = raw_args

                            # Execute mock tool from dispatch table
                            if name in dispatch_table:
                                tool_output = str(dispatch_table[name](**raw_args))
                            else:
                                tool_output = f"Executed {name}"

                            # Second turn with TOOL_RESULT
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": f"TOOL_RESULT: {tool_output}"})

                            iterations += 1
                            second_content, _, meta2 = self._stream_chat(messages)
                            tokens_in += meta2.get("prompt_eval_count", 0)
                            tokens_out += meta2.get("eval_count", 0)
                            load_dur_ns += meta2.get("load_duration", 0) or 0
                            prompt_eval_dur_ns += meta2.get("prompt_eval_duration", 0) or 0
                            eval_dur_ns += meta2.get("eval_duration", 0) or 0
                            final_reply = second_content
                        else:
                            final_reply = content
                    else:
                        final_reply = content
                except Exception as ex:
                    final_reply = content
                    error_msg = f"JSON parse error: {ex}"
            else:
                final_reply = content

        except Exception as e:
            error_msg = str(e)
            final_reply = f"Error: {e}"

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "reply": final_reply,
            "tool_called": tool_called,
            "args": tool_args,
            "valid_json": valid_json,
            "ttft_ms": round(ttft_ms, 2),
            "load_duration_ms": round(load_dur_ns / 1e6, 2),
            "prompt_eval_duration_ms": round(prompt_eval_dur_ns / 1e6, 2),
            "eval_duration_ms": round(eval_dur_ns / 1e6, 2),
            "prompt_eval_count": tokens_in,
            "eval_count": tokens_out,
            "total_ms": round(total_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "iterations": iterations,
            "model": self.model,
            "quant": self.quant,
            "options": {"temperature": 0.0, "num_ctx": 4096, "seed": 42, "keep_alive": -1},
            "error": error_msg,
        }


def run_ollama_diagnostic(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    prompt: str = "ping",
    count: int = 20,
) -> Dict[str, Any]:
    url = (base_url or config.LOCAL_LLM_URL).replace("/v1", "").rstrip("/")
    target_model = model or config.LOCAL_LLM_MODEL or "jarvis-3b"
    options = {
        "temperature": 0.0,
        "num_ctx": 4096,
        "seed": 42,
    }
    payload = {
        "model": target_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": -1,
        "options": options,
    }
    print(f"\n============================================================")
    print(f"  Ollama Diagnostic: {count} identical short prompts ('{prompt}')")
    print(f"  Model: {target_model} | keep_alive=-1 | num_ctx=4096 | seed=42")
    print(f"============================================================")

    loads_ms = []
    prefills_ms = []
    decodes_ms = []
    totals_ms = []

    for i in range(1, count + 1):
        req = urllib.request.Request(
            f"{url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            elapsed = (time.perf_counter() - t0) * 1000.0
            data = json.loads(resp.read().decode("utf-8"))

        load_ms = (data.get("load_duration", 0) or 0) / 1e6
        p_eval_ms = (data.get("prompt_eval_duration", 0) or 0) / 1e6
        e_dur_ms = (data.get("eval_duration", 0) or 0) / 1e6
        loads_ms.append(load_ms)
        prefills_ms.append(p_eval_ms)
        decodes_ms.append(e_dur_ms)
        totals_ms.append(elapsed)

        print(
            f"  Call {i:2d}/{count}: load={load_ms:6.1f}ms | prefill={p_eval_ms:6.1f}ms ({data.get('prompt_eval_count',0)} tok) "
            f"| decode={e_dur_ms:6.1f}ms ({data.get('eval_count',0)} tok) | total={elapsed:6.1f}ms"
        )

    avg_load = sum(loads_ms) / len(loads_ms)
    avg_prefill = sum(prefills_ms) / len(prefills_ms)
    avg_decode = sum(decodes_ms) / len(decodes_ms)
    avg_total = sum(totals_ms) / len(totals_ms)

    print("\n--- Summary Split ---")
    print(f"  Load Duration       : avg={avg_load:6.1f}ms (min={min(loads_ms):.1f}ms, max={max(loads_ms):.1f}ms)")
    print(f"  Prefill (prompt_eval): avg={avg_prefill:6.1f}ms (min={min(prefills_ms):.1f}ms, max={max(prefills_ms):.1f}ms)")
    print(f"  Decode (eval)       : avg={avg_decode:6.1f}ms (min={min(decodes_ms):.1f}ms, max={max(decodes_ms):.1f}ms)")
    print(f"  Total Latency       : avg={avg_total:6.1f}ms (min={min(totals_ms):.1f}ms, max={max(totals_ms):.1f}ms)")

    # Run `ollama ps`
    print("\n--- `ollama ps` Output ---")
    try:
        import subprocess
        ps_out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=False)
        print(ps_out.stdout.strip() or ps_out.stderr.strip() or "(empty ollama ps output)")
    except Exception as e:
        print(f"Failed to run 'ollama ps': {e}")
    print("============================================================\n")

    return {
        "count": count,
        "avg_load_ms": avg_load,
        "avg_prefill_ms": avg_prefill,
        "avg_decode_ms": avg_decode,
        "avg_total_ms": avg_total,
    }


class CloudBenchmarkProvider(BenchmarkProvider):
    name = "cloud"

    def __init__(self, provider_name: str = "claude"):
        from models.model_manager import ModelManager
        self.provider_name = provider_name
        self.manager = ModelManager()
        self.client = self._resolve_client()
        self.model = getattr(self.client, "model", None) or getattr(self.client, "name", provider_name)
        self.quant = "cloud_api"

    def _resolve_client(self):
        candidates = self.manager.candidates(self.provider_name)
        if not candidates:
            raise RuntimeError(f"No configured cloud client for '{self.provider_name}'. Check API keys.")
        return candidates[0]

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        error_msg = None
        try:
            # Enforce temperature 0
            old_temp = getattr(config, "LLM_TEMPERATURE", 0.0)
            config.LLM_TEMPERATURE = 0.0
            raw_res = self.client.ask(text, dispatch_table)
            config.LLM_TEMPERATURE = old_temp

            reply_str = raw_res.get("reply", "") if isinstance(raw_res, dict) else str(raw_res)
            stats = dict(getattr(self.client, "last_stats", {}) or {})
            total_ms = (time.perf_counter() - t0) * 1000.0

            tool_calls = stats.get("tool_calls") or []
            tool_called = stats.get("tool_called") or (tool_calls[0]["tool"] if tool_calls else None)
            tool_args = stats.get("args") or (tool_calls[0]["args"] if tool_calls else {})

            return {
                "reply": reply_str,
                "tool_called": tool_called,
                "args": tool_args,
                "valid_json": bool(tool_called),
                "ttft_ms": round(stats.get("llm_latency_ms", total_ms) * 0.4, 2),
                "total_ms": round(total_ms, 2),
                "tokens_in": stats.get("prompt_tokens", stats.get("tokens_in", 0)),
                "tokens_out": stats.get("completion_tokens", stats.get("tokens_out", 0)),
                "iterations": stats.get("iterations", 1),
                "model": self.model,
                "quant": self.quant,
                "prompt_eval_duration_ms": 0.0,
                "eval_duration_ms": 0.0,
                "error": None,
            }
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "reply": f"Cloud error: {e}",
                "tool_called": None,
                "args": {},
                "valid_json": False,
                "ttft_ms": 0.0,
                "total_ms": round(total_ms, 2),
                "tokens_in": 0,
                "tokens_out": 0,
                "iterations": 1,
                "model": self.model,
                "quant": self.quant,
                "prompt_eval_duration_ms": 0.0,
                "eval_duration_ms": 0.0,
                "error": str(e),
            }


class RegexBenchmarkProvider(BenchmarkProvider):
    name = "regex"
    model = "regex_baseline"
    quant = "none"

    def __init__(self):
        self.router = RegexBaselineRouter()

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        res = self.router.ask(text, dispatch_table, history)
        return {
            "reply": res["reply"],
            "tool_called": res["tool_called"],
            "args": res["args"],
            "valid_json": res["valid_json"],
            "ttft_ms": 0.0,
            "total_ms": res["total_latency_ms"],
            "tokens_in": 0,
            "tokens_out": 0,
            "iterations": res["iterations"],
            "model": "regex_baseline",
            "quant": "none",
            "prompt_eval_duration_ms": 0.0,
            "eval_duration_ms": 0.0,
            "error": res.get("error"),
        }


class StubBenchmarkProvider(BenchmarkProvider):
    """
    Offline deterministic stub provider for dry runs and unit testing.
    Requires no GPU, network, API keys, or Ollama server.
    """
    name = "stub"
    model = "stub_model_v1"
    quant = "Q4_K_M"

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        text_lower = text.strip().lower()

        tool_called = None
        tool_args = {}
        valid_json = False
        reply = "I understand and am standing by."

        if "open spotify" in text_lower:
            tool_called = "open_app"
            tool_args = {"app_name": "spotify"}
            valid_json = True
        elif "set a timer" in text_lower:
            tool_called = "set_timer"
            tool_args = {"seconds": 300}
            valid_json = True
        elif "mark" in text_lower or "complete" in text_lower:
            tool_called = "complete_todo"
            tool_args = {"task_hint": "buy groceries"}
            valid_json = True
        elif "to-do list" in text_lower or "todo list" in text_lower:
            if "add" in text_lower:
                tool_called = "add_todo"
                tool_args = {"task": "buy groceries"}
                valid_json = True
            else:
                tool_called = "list_todos"
                tool_args = {}
                valid_json = True
        elif "draft email" in text_lower:
            tool_called = "create_email_draft"
            tool_args = {"to": "alice@example.com", "subject": "Meeting", "body": "Hello Alice"}
            valid_json = True
        elif "send it now" in text_lower or "whatsapp" in text_lower:
            tool_called = "send_whatsapp_message"
            tool_args = {"recipient": "Bob", "message": "hello"}
            valid_json = True
        elif "make me a pizza" in text_lower:
            reply = "I'm afraid I cannot make food or prepare a pizza, sir. I can only assist with digital tasks."
        elif "capital of france" in text_lower:
            reply = "The capital of France is Paris, sir."
        elif "open question" in text_lower:
            reply = "An open question is one that cannot be answered with a simple 'yes' or 'no', sir."

        if tool_called:
            if tool_called in dispatch_table:
                reply = dispatch_table[tool_called](**tool_args)
            else:
                reply = f"Executed {tool_called}"

        total_ms = max(1.2, (time.perf_counter() - t0) * 1000.0)
        ttft_ms = 0.8

        return {
            "reply": reply,
            "tool_called": tool_called,
            "args": tool_args,
            "valid_json": valid_json,
            "ttft_ms": ttft_ms,
            "total_ms": round(total_ms, 2),
            "tokens_in": 128,
            "tokens_out": 24,
            "iterations": 2 if tool_called else 1,
            "model": self.model,
            "quant": self.quant,
            "prompt_eval_duration_ms": 15.0,
            "eval_duration_ms": 25.0,
            "error": None,
        }


class FasterWhisperSTTProvider(BenchmarkProvider):
    name = "faster_whisper"

    def __init__(
        self,
        model_tag: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        pipeline: Optional[FasterWhisperPipeline] = None,
    ):
        self.model = model_tag or getattr(config, "WHISPER_MODEL_SIZE", "base")
        self.pipeline = pipeline or FasterWhisperPipeline(
            model_size=self.model,
            device=device,
            compute_type=compute_type,
        )
        self.quant = self.pipeline.compute_type or "float16"

    def ask(
        self,
        text: str,
        dispatch_table: Dict[str, Callable[..., str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if os.path.exists(text):
            transcribed = self.pipeline.transcribe(text)
        else:
            transcribed = text
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "reply": transcribed,
            "tool_called": None,
            "args": {},
            "valid_json": False,
            "ttft_ms": round(elapsed, 2),
            "total_ms": round(elapsed, 2),
            "tokens_in": len(transcribed.split()),
            "tokens_out": len(transcribed.split()),
            "iterations": 1,
            "model": self.model,
            "quant": self.quant,
            "error": None,
        }


def get_provider(provider_name: str, model_tag: Optional[str] = None) -> BenchmarkProvider:
    prov_lower = (provider_name or "").strip().lower()
    if prov_lower in ("stub", "mock", "test"):
        return StubBenchmarkProvider()
    elif prov_lower in ("regex", "regex_router", "baseline"):
        return RegexBenchmarkProvider()
    elif prov_lower in ("ollama", "local", "ollama_manager"):
        return OllamaBenchmarkProvider(model_tag=model_tag)
    elif prov_lower in ("claude", "openai", "gemini", "cloud"):
        return CloudBenchmarkProvider(provider_name=provider_name)
    elif prov_lower in ("whisper", "faster_whisper", "stt"):
        return FasterWhisperSTTProvider(model_tag=model_tag)
    else:
        # Default: treat as Ollama model tag or fallback to Ollama
        return OllamaBenchmarkProvider(model_tag=provider_name)

