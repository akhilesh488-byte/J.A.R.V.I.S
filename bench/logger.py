"""
Structured CSV Logger for benchmark experiment runs.
Validates CSV headers and appends run records.
"""

import csv
import os
from typing import Any, Dict, List

REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "text",
    "regex_matched",
    "provider",
    "model",
    "total_ms",
    "llm_ms",
    "tool_ms",
    "tokens_in",
    "tokens_out",
    "tool_called",
    "valid_json",
    "exact_match",
    "error",
    "escalated",
]


def log_experiment_run(csv_path: str, data_row: Dict[str, Any]) -> None:
    """
    Appends a row of experiment data to csv_path with header validation.
    If the file does not exist, writes the header first.
    If the file exists and is non-empty, verifies the header matches REQUIRED_COLUMNS.
    """
    abs_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    file_exists = os.path.exists(abs_path)
    file_empty = os.path.getsize(abs_path) == 0 if file_exists else True

    if file_exists and not file_empty:
        with open(abs_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or [c.strip() for c in header] != REQUIRED_COLUMNS:
                raise ValueError(
                    f"CSV header validation failed for {csv_path}. "
                    f"Expected: {REQUIRED_COLUMNS}, got: {header}"
                )

    row_data = {}
    for col in REQUIRED_COLUMNS:
        val = data_row.get(col, "")
        if val is None:
            val = ""
        elif isinstance(val, bool):
            val = str(val)
        row_data[col] = val

    with open(abs_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        if file_empty:
            writer.writeheader()
        writer.writerow(row_data)
