"""
Captures hardware, OS, and environment metadata for benchmark runs.
Writes results to results/env.json:
- nvidia-smi output
- OS build and architecture
- Python version and pip freeze packages
- git commit SHA
- SHA-256 of system prompt template, tool schema, and dataset file
- Windows power plan scheme
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def file_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                fp = os.path.join(root, f)
                h.update(f.encode("utf-8"))
                try:
                    with open(fp, "rb") as subf:
                        while chunk := subf.read(65536):
                            h.update(chunk)
                except Exception:
                    pass
        return h.hexdigest()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def capture_environment(
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    env_data: Dict[str, Any] = {}

    # OS Build and Python Version
    env_data["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
    }
    env_data["python"] = {
        "version": sys.version,
        "executable": sys.executable,
    }

    # Git commit
    try:
        git_res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        env_data["git_commit"] = git_res.stdout.strip()
    except Exception as e:
        env_data["git_commit"] = f"unknown ({e})"

    # nvidia-smi
    try:
        smi_res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        env_data["nvidia_smi"] = smi_res.stdout.strip() if smi_res.returncode == 0 else smi_res.stderr.strip()
    except Exception as e:
        env_data["nvidia_smi"] = f"unavailable ({e})"

    # Power plan (Windows)
    try:
        power_res = subprocess.run("powercfg /getactivescheme", shell=True, capture_output=True, text=True)
        env_data["power_plan"] = power_res.stdout.strip() if power_res.returncode == 0 else power_res.stderr.strip()
    except Exception as e:
        env_data["power_plan"] = f"unknown ({e})"

    # Pip freeze
    try:
        pip_res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        env_data["pip_freeze"] = [pkg.strip() for pkg in pip_res.stdout.splitlines() if pkg.strip()]
    except Exception as e:
        env_data["pip_freeze"] = [f"error ({e})"]

    # SHA-256 of artifacts
    prompt_template_path = os.path.join(ROOT, "jarvis_data", "system_prompt_template.txt")
    if not os.path.exists(prompt_template_path):
        prompt_template_path = os.path.join(ROOT, "core", "prompt.py")

    tool_lines_path = os.path.join(ROOT, "jarvis_data", "tool_lines.txt")
    tool_registry_path = os.path.join(ROOT, "tools", "tool_registry.py")

    env_data["hashes"] = {
        "system_prompt_template_sha256": file_sha256(prompt_template_path),
        "tool_lines_sha256": file_sha256(tool_lines_path) if os.path.exists(tool_lines_path) else None,
        "tool_registry_sha256": file_sha256(tool_registry_path),
        "dataset_sha256": file_sha256(dataset_path) if dataset_path else None,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(env_data, f, indent=2)

    return env_data


if __name__ == "__main__":
    out = os.path.join(ROOT, "results", "env.json")
    capture_environment(output_path=out)
    print(f"Environment captured to {out}")
