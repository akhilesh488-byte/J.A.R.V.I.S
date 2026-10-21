"""
Dataset loader for benchmarking Jarvis.
Loads and validates rows matching the schema:
- id: str
- category: str (simple, parametric, reasoning, adversarial, unsupported, multi_turn, etc.)
- split: str (dev, test, sample)
- text: str (the user command or query)
- audio_path: Optional[str]
- history: List[dict] (spoken-only turns: [{"role": "user"|"assistant", "content": "..."}, ...])
- expected_tool: Optional[str] (registry tool name or null)
- expected_args: Optional[dict] (expected kwargs dictionary or null)
- expected_reply_type: Optional[str] (tool_call, conversational, unsupported, clarification)

Supports loading from both JSONL files and CSV template files (like jarvis_test_set_template.csv).
"""

from __future__ import annotations

import csv
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
        exp_tool = data.get("expected_tool")
        if exp_tool is not None and str(exp_tool).strip().lower() in ("", "none", "null"):
            exp_tool = None

        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "general")),
            split=str(data.get("split", "sample")),
            text=str(data.get("text", "")),
            audio_path=data.get("audio_path"),
            history=clean_hist,
            expected_tool=exp_tool,
            expected_args=data.get("expected_args") or {},
            expected_reply_type=data.get("expected_reply_type") or data.get("expected_behavior"),
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


def parse_csv_template(csv_path: str) -> List[DatasetRow]:
    """
    Parse CSV rows from jarvis_test_set_template.csv into DatasetRow objects.
    Extracts {id, category, history (spoken-only turns), text, expected_tool (or null), expected_args}.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[DatasetRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd_id = row.get("id", "").strip()
            if not cmd_id:
                continue

            category = row.get("category", "").strip() or "tool"
            text = (row.get("text") or "").strip()
            if not text:
                # If text is empty in template, check hint or target_tool
                text = (row.get("hint") or "").strip()
                if not text and row.get("target_tool"):
                    text = f"open {row['target_tool']}" if row["target_tool"] == "open_app" else row["target_tool"]

            # Parse spoken-only history turns
            history: List[Dict[str, str]] = []
            earlier_u = (row.get("earlier_user") or "").strip()
            earlier_j = (row.get("earlier_jarvis") or "").strip()
            if earlier_u:
                history.append({"role": "user", "content": earlier_u})
            if earlier_j:
                history.append({"role": "assistant", "content": earlier_j})
            if not history and row.get("history"):
                try:
                    raw_h = json.loads(row["history"])
                    if isinstance(raw_h, list):
                        history = [
                            {"role": h.get("role", "user"), "content": str(h.get("content", ""))}
                            for h in raw_h
                            if isinstance(h, dict) and h.get("role") in ("user", "assistant")
                        ]
                except Exception:
                    pass

            # Parse expected tool
            exp_tool = (row.get("expected_tool") or row.get("target_tool") or "").strip()
            if not exp_tool or exp_tool.lower() in ("none", "null"):
                exp_tool = None

            # Parse expected args
            args_raw = (row.get("expected_args_json") or row.get("expected_args") or "").strip()
            expected_args = {}
            if args_raw:
                try:
                    parsed_args = json.loads(args_raw)
                    if isinstance(parsed_args, dict):
                        expected_args = parsed_args
                except Exception:
                    expected_args = {}

            behavior = (row.get("expected_behavior") or row.get("expected_reply_type") or "").strip() or None

            dataset_row = DatasetRow(
                id=cmd_id,
                category=category,
                split="test",
                text=text,
                audio_path=None,
                history=history,
                expected_tool=exp_tool,
                expected_args=expected_args,
                expected_reply_type=behavior,
            )
            rows.append(dataset_row)

    return rows


def convert_template_csv_to_jsonl(csv_path: str, jsonl_out_path: str) -> List[DatasetRow]:
    """
    Converts jarvis_test_set_template.csv into a clean JSONL dataset file.
    """
    rows = parse_csv_template(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_out_path)), exist_ok=True)
    with open(jsonl_out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")
    return rows


def load_dataset(path: str) -> List[DatasetRow]:
    """
    Load dataset rows from either a JSONL file or a CSV template file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.lower().endswith(".csv"):
        return parse_csv_template(path)

    # Otherwise load as JSONL
    rows: List[DatasetRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                item = json.loads(line_str)
                rows.append(DatasetRow.from_dict(item))
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
    return rows
