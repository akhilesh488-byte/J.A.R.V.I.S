"""
Dataset loader for benchmarking Jarvis.
Loads and validates JSONL rows formatted with:
- id: str
- category: str (simple, parametric, reasoning, adversarial, unsupported, multi_turn)
- split: str (dev, test, sample)
- text: str (the user command or query)
- audio_path: Optional[str] (pre-recorded audio if benchmarking STT + LLM)
- history: List[dict] (spoken-only turns: [{"role": "user"|"assistant", "content": "..."}, ...])
- expected_tool: Optional[str] (registry tool name or null)
- expected_args: Optional[dict] (expected kwargs dictionary or null)
- expected_reply_type: Optional[str] (tool_call, conversational, unsupported, clarification)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetRow:
    id: str
    category: str
    split: str
    text: str
    audio_path: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    expected_tool: Optional[str] = None
    expected_args: Dict[str, Any] = field(default_factory=dict)
    expected_reply_type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> DatasetRow:
        hist = data.get("history") or []
        # Ensure history only retains spoken turns (role in user/assistant)
        clean_hist = [
            {"role": h.get("role", "user"), "content": str(h.get("content", ""))}
            for h in hist
            if isinstance(h, dict) and h.get("role") in ("user", "assistant")
        ]
        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "general")),
            split=str(data.get("split", "sample")),
            text=str(data.get("text", "")),
            audio_path=data.get("audio_path"),
            history=clean_hist,
            expected_tool=data.get("expected_tool"),
            expected_args=data.get("expected_args") or {},
            expected_reply_type=data.get("expected_reply_type"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "split": self.split,
            "text": self.text,
            "audio_path": self.audio_path,
            "history": self.history,
            "expected_tool": self.expected_tool,
            "expected_args": self.expected_args,
            "expected_reply_type": self.expected_reply_type,
        }


def load_dataset(jsonl_path: str) -> List[DatasetRow]:
    """
    Load dataset rows from a JSONL file.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    rows: List[DatasetRow] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                item = json.loads(line_str)
                rows.append(DatasetRow.from_dict(item))
            except Exception as e:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_num}: {e}") from e
    return rows


def load_audio_directory(audio_dir: str) -> List[DatasetRow]:
    """
    Ingest live audio/voice recordings from a designated custom audio directory.
    Supports raw .wav files directly, with optional metadata/references if present.
    """
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Custom audio directory not found: {audio_dir}")

    # Check for optional reference/metadata files
    refs = {}
    ref_file = os.path.join(audio_dir, "references.json")
    if os.path.exists(ref_file):
        try:
            with open(ref_file, "r", encoding="utf-8") as rf:
                refs = json.load(rf)
        except Exception:
            refs = {}

    rows: List[DatasetRow] = []
    wav_files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])
    if not wav_files:
        raise ValueError(f"No .wav audio recordings found in designated directory: {audio_dir}")

    for idx, f in enumerate(wav_files, start=1):
        full_path = os.path.join(audio_dir, f)
        base_id = os.path.splitext(f)[0]
        ref_info = refs.get(base_id, {}) if isinstance(refs, dict) else {}
        expected_text = ref_info.get("reference") or ref_info.get("text") or ""
        expected_tool = ref_info.get("expected_tool")
        expected_args = ref_info.get("expected_args") or {}
        category = ref_info.get("category", "live_audio")

        rows.append(
            DatasetRow(
                id=f"audio_{idx:03d}_{base_id}",
                category=category,
                split="live_audio",
                text=expected_text,
                audio_path=full_path,
                expected_tool=expected_tool,
                expected_args=expected_args,
                expected_reply_type=ref_info.get("expected_reply_type", "tool_call"),
            )
        )
    return rows

