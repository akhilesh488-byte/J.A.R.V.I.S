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
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_gguf_path() -> Optional[str]:
    """Locates the GGUF model file if available locally."""
    env_p = os.environ.get("GGUF_PATH")
    if env_p and os.path.exists(env_p):
        return env_p
    default_p = os.path.join(ROOT, "Llama-3.2-3B-Instruct.Q4_K_M.gguf")
    if os.path.exists(default_p):
        return default_p
    try:
        for fname in os.listdir(ROOT):
            if fname.lower().endswith(".gguf"):
                return os.path.join(ROOT, fname)
    except Exception:
        pass
    return None


def get_heavy_running_processes() -> List[Dict[str, str]]:
    """Detects heavy background processes that may interfere with benchmarks."""
    heavy_names = {
        "code.exe", "chrome.exe", "msedge.exe", "firefox.exe", "devenv.exe",
        "pycharm64.exe", "idea64.exe", "discord.exe", "slack.exe", "spotify.exe",
        "steam.exe", "blender.exe", "obs64.exe", "zoom.exe", "teams.exe"
    }
    found = []
    try:
        res = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=True)
        import csv as _csv
        import io as _io
        reader = _csv.reader(_io.StringIO(res.stdout))
        for row in reader:
            if row and len(row) >= 2:
                name = row[0].strip()
                pid = row[1].strip()
                mem = row[4].strip() if len(row) >= 5 else ""
                if name.lower() in heavy_names:
                    found.append({"name": name, "pid": pid, "memory": mem})
    except Exception:
        pass
    return found


def capture_environment(
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    model_tag: Optional[str] = None,
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

    # Model and GGUF info
    gguf_path = find_gguf_path()
    gguf_hash = file_sha256(gguf_path) if gguf_path else None
    tag = model_tag or os.environ.get("LOCAL_LLM_MODEL") or os.environ.get("OLLAMA_MODEL", "jarvis-3b")
    env_data["ollama_tag"] = tag
    env_data["gguf_path"] = gguf_path

    # Whisper configuration
    import config
    env_data["whisper"] = {
        "model": getattr(config, "WHISPER_MODEL_SIZE", "base"),
        "device": getattr(config, "WHISPER_DEVICE", "cuda"),
        "compute_type": getattr(config, "WHISPER_COMPUTE_TYPE", "float16"),
    }

    # Heavy running processes
    env_data["heavy_running_processes"] = get_heavy_running_processes()

    # SHA-256 of artifacts
    prompt_template_path = os.path.join(ROOT, "jarvis_data", "system_prompt_template.txt")
    if not os.path.exists(prompt_template_path):
        prompt_template_path = os.path.join(ROOT, "core", "prompt.py")

    tool_lines_path = os.path.join(ROOT, "jarvis_data", "tool_lines.txt")
    tool_registry_path = os.path.join(ROOT, "tools", "tool_registry.py")

    prompt_sha = file_sha256(prompt_template_path)
    tool_schema_sha = file_sha256(tool_lines_path) if os.path.exists(tool_lines_path) else file_sha256(tool_registry_path)
    test_file_sha = file_sha256(dataset_path) if dataset_path else None

    env_data["hashes"] = {
        "gguf_sha256": gguf_hash,
        "ollama_tag": tag,
        "prompt_sha256": prompt_sha,
        "tool_schema_sha256": tool_schema_sha,
        "test_file_sha256": test_file_sha,
        "system_prompt_template_sha256": prompt_sha,
        "tool_lines_sha256": file_sha256(tool_lines_path) if os.path.exists(tool_lines_path) else None,
        "tool_registry_sha256": file_sha256(tool_registry_path),
        "tool_results_sha256": file_sha256(os.path.join(ROOT, "tool_results.json")),
        "dataset_sha256": test_file_sha,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(env_data, f, indent=2)

    return env_data


def compare_runs(run1_path: str, run2_path: str) -> Dict[str, Any]:
    """
    Compares two benchmark runs by their environment metadata / hashes.
    Refuses comparison if critical hashes (prompt, tool_schema, test_file, or gguf) differ.
    """
    def _find_env(path: str) -> Dict[str, Any]:
        if os.path.isdir(path):
            env_file = os.path.join(path, "env.json")
        else:
            env_file = os.path.join(os.path.dirname(os.path.abspath(path)), "env.json")
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"env.json not found for run path: {path}")

    env1 = _find_env(run1_path)
    env2 = _find_env(run2_path)

    h1 = env1.get("hashes", {})
    h2 = env2.get("hashes", {})

    critical_keys = ["prompt_sha256", "tool_schema_sha256", "test_file_sha256", "gguf_sha256"]
    mismatches = []
    for k in critical_keys:
        v1 = h1.get(k)
        v2 = h2.get(k)
        if v1 and v2 and v1 != v2:
            mismatches.append(f"{k} mismatch: {v1[:12]}... != {v2[:12]}...")

    if mismatches:
        err = f"Refusing to compare runs with different environment hashes: {'; '.join(mismatches)}"
        raise ValueError(err)

    return {"status": "MATCH", "env1": env1, "env2": env2}


if __name__ == "__main__":
    out = os.path.join(ROOT, "results", "env.json")
    capture_environment(output_path=out)
    print(f"Environment captured to {out}")
