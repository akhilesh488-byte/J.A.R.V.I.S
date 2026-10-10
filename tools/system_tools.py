"""
System-level tools: opening apps, taking screenshots.

Everything here is deliberately terminal/subprocess based rather than
GUI-automation based (no xdotool/pyautogui), so it works the same way
whether you're on Windows, X11, or Wayland.
"""

import os
import platform
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _current_os() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    return "linux"


import re


def open_app(app_name: str) -> str:
    """
    Open an application by its voice-friendly name.
    Looks up config.APP_ALIASES first; falls back to trying the raw
    name as a shell command if there's no alias.
    """
    os_key = _current_os()
    # Strip punctuation, leading "the", "up", "a", "an", and trailing "app"
    clean = re.sub(r"[^\w\s-]", "", app_name).strip().lower()
    clean = re.sub(r"^(?:the|up|a|an)\s+", "", clean)
    clean = re.sub(r"\s+app$", "", clean)
    clean = re.sub(r"^(?:the|up|a|an)\s+", "", clean)

    alias_entry = config.APP_ALIASES.get(clean)
    if not alias_entry:
        alias_entry = config.APP_ALIASES.get(clean.replace(" ", ""))

    command = alias_entry[os_key] if alias_entry else clean
    display_name = clean.title() if clean else app_name.strip()

    try:
        if os_key == "windows":
            # 1. URL or URI protocol (e.g. spotify:, https://)
            if command.startswith(("http://", "https://")) or command.endswith(":"):
                os.startfile(command)
                return f"Opening {display_name}."

            # 2. Try os.startfile for registered Windows executables and app aliases
            try:
                os.startfile(command)
                return f"Opening {display_name}."
            except Exception:
                pass

            # 3. Fallback to start command with empty title parameter
            subprocess.Popen(f'start "" {command}', shell=True)
            return f"Opening {display_name}."
        else:
            subprocess.Popen(command, shell=True)
            return f"Opening {display_name}."
    except Exception as e:
        return f"Couldn't open {display_name}: {e}"


def take_screenshot(save_path: str = None) -> str:
    """
    Takes a screenshot and saves it to disk. Returns the file path.
    Uses 'mss' which works on Windows and X11. On Wayland (GNOME),
    mss can't capture the screen directly due to the compositor's
    security model — this will raise, and the caller should fall back
    to telling the user to switch to X11 or use a portal-based tool
    (e.g. `gnome-screenshot`, which we shell out to as a fallback below).
    """
    if save_path is None:
        save_path = os.path.join(config.DATA_DIR, "screenshot.png")

    try:
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=save_path)
        return save_path
    except Exception:
        # Fallback for Wayland/GNOME: shell out to gnome-screenshot if present
        try:
            subprocess.run(["gnome-screenshot", "-f", save_path], check=True)
            return save_path
        except Exception as e:
            raise RuntimeError(
                "Screen capture failed. On GNOME/Wayland, install "
                "gnome-screenshot (`sudo apt install gnome-screenshot`), "
                f"or switch your session to X11. Original error: {e}"
            )
