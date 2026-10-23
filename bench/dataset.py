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

Supports loading from JSONL files and CSV template files (like jarvis_test_set_template.csv).
Separates datasets strictly into 'dev' and 'test' groups with independent execution.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
            split=str(data.get("split", "test")),
            text=str(data.get("text", "")),
            audio_path=data.get("audio_path"),
            history=clean_hist,
            expected_tool=exp_tool,
            expected_args=data.get("expected_args") or {},
            expected_reply_type=data.get("expected_reply_type") or data.get("expected_behavior"),
        )

    @property
    def has_label(self) -> bool:
        if self.expected_tool is not None:
            return True
        if self.expected_reply_type is not None and str(self.expected_reply_type).strip().lower() not in ("", "none", "null"):
            return True
        if self.expected_args:
            return True
        if self.split in ("dev", "sample"):
            return True
        if self.category and self.category.lower() not in ("", "none", "null", "unlabeled", "unknown", "tool"):
            return True
        return False

    @property
    def expected_behavior(self) -> Optional[str]:
        return self.expected_reply_type

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
            "expected_behavior": self.expected_behavior,
            "expected_reply_type": self.expected_reply_type,
            "has_label": self.has_label,
        }


def load_dev60_labels(labels_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Loads ground truth tool and arg definitions from dev60_proposed_labels.csv."""
    path = labels_path or os.path.join(ROOT, "dev60_proposed_labels.csv")
    if not os.path.exists(path):
        return {}
    labels: Dict[str, Dict[str, str]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("id") or "").strip()
                if cid:
                    labels[cid] = row
    except Exception:
        pass
    return labels


def parse_csv_template(csv_path: str, split: Optional[str] = None) -> List[DatasetRow]:
    """
    Parse CSV rows from jarvis_test_set_template.csv into DatasetRow objects.
    Injects ground truth expected_tool and expected_args for dev commands.
    Filters by split ('dev' or 'test') when requested.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dev60_map = load_dev60_labels()

    rows: List[DatasetRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd_id = row.get("id", "").strip()
            if not cmd_id:
                continue

            # Determine split
            row_split = (row.get("split") or "").strip().lower()
            if not row_split:
                row_split = "dev" if cmd_id in dev60_map else "test"

            if split and row_split != split.lower():
                continue

            dev_info = dev60_map.get(cmd_id, {})

            category = (dev_info.get("proposed_category") or row.get("category") or "").strip() or "tool"
            text = (row.get("text") or "").strip()
            if not text:
                text = (dev_info.get("transcript_seen") or row.get("hint") or "").strip()
                if not text and row.get("target_tool"):
                    text = f"open {row['target_tool']}" if row["target_tool"] == "open_app" else row["target_tool"]

            # Parse spoken-only history turns
            history: List[Dict[str, str]] = []
            earlier_u = (dev_info.get("earlier_user") or row.get("earlier_user") or "").strip()
            earlier_j = (dev_info.get("earlier_jarvis") or row.get("earlier_jarvis") or "").strip()
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
            exp_tool = (
                dev_info.get("expected_tool")
                if "expected_tool" in dev_info
                else (row.get("expected_tool") or row.get("target_tool") or "")
            )
            exp_tool = str(exp_tool).strip() if exp_tool else ""
            if not exp_tool or exp_tool.lower() in ("none", "null"):
                exp_tool = None

            # Parse expected args
            args_raw = (
                dev_info.get("expected_args_json")
                if "expected_args_json" in dev_info
                else (row.get("expected_args_json") or row.get("expected_args") or "")
            )
            args_raw = str(args_raw).strip() if args_raw else ""
            expected_args = {}
            if args_raw:
                try:
                    parsed_args = json.loads(args_raw)
                    if isinstance(parsed_args, dict):
                        expected_args = parsed_args
                except Exception:
                    expected_args = {}

            behavior = (
                dev_info.get("proposed_behavior")
                or row.get("expected_behavior")
                or row.get("expected_reply_type")
                or ""
            ).strip() or None

            dataset_row = DatasetRow(
                id=cmd_id,
                category=category,
                split=row_split,
                text=text,
                audio_path=None,
                history=history,
                expected_tool=exp_tool,
                expected_args=expected_args,
                expected_reply_type=behavior,
            )
            rows.append(dataset_row)

    return rows


def convert_template_csv_to_jsonl(csv_path: str, jsonl_out_path: str, split: Optional[str] = None) -> List[DatasetRow]:
    """
    Converts CSV template into a clean JSONL dataset file, optionally filtered by split.
    """
    rows = parse_csv_template(csv_path, split=split)
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_out_path)), exist_ok=True)
    with open(jsonl_out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")
    return rows


def generate_isolated_datasets():
    """Generates isolated dev_set.jsonl and test_set.jsonl files."""
    template_path = os.path.join(ROOT, "jarvis_test_set_template.csv")
    dev_path = os.path.join(ROOT, "bench", "data", "dev_set.jsonl")
    test_path = os.path.join(ROOT, "bench", "data", "test_set.jsonl")
    convert_template_csv_to_jsonl(template_path, dev_path, split="dev")
    convert_template_csv_to_jsonl(template_path, test_path, split="test")


def load_dataset(path: str, split: Optional[str] = None) -> List[DatasetRow]:
    """
    Load dataset rows from either a JSONL file or a CSV template file,
    with optional filtering by split ('dev' or 'test').
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.lower().endswith(".csv"):
        return parse_csv_template(path, split=split)

    # Otherwise load as JSONL
    rows: List[DatasetRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                item = json.loads(line_str)
                row_split = (item.get("split") or "").strip().lower()
                if split and row_split and row_split != split.lower():
                    continue
                rows.append(DatasetRow.from_dict(item))
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
    return rows


def load_dev_set() -> List[DatasetRow]:
    """Loads the active 60-command dev benchmark set."""
    dev_jsonl = os.path.join(ROOT, "bench", "data", "dev_set.jsonl")
    if os.path.exists(dev_jsonl):
        return load_dataset(dev_jsonl, split="dev")
    return load_dataset(os.path.join(ROOT, "jarvis_test_set_template.csv"), split="dev")


def load_test_set() -> List[DatasetRow]:
    """Loads the 60-command test benchmark set."""
    test_jsonl = os.path.join(ROOT, "bench", "data", "test_set.jsonl")
    if os.path.exists(test_jsonl):
        return load_dataset(test_jsonl, split="test")
    return load_dataset(os.path.join(ROOT, "jarvis_test_set_template.csv"), split="test")
