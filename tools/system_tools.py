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


def open_app(app_name: str) -> str:
    """
    Open an application by its voice-friendly name.
    Looks up config.APP_ALIASES first; falls back to trying the raw
    name as a shell command if there's no alias.

    Windows notes:
    - If the command looks like a URI (contains ":" or "/"), or is a
      multi-word shell command, we use 'start' via subprocess so the
      shell can resolve it.
    - For single-word names (e.g. "code", "explorer"), os.startfile()
      / ShellExecute is used; it searches PATH *and* app registrations,
      so registered apps like Notepad, Calculator, etc. all work.
    """
    os_key = _current_os()
    name = app_name.strip().lower()

    alias_entry = config.APP_ALIASES.get(name)
    command = alias_entry[os_key] if alias_entry else name

    try:
        if os_key == "windows":
            # URIs (spotify:, shell:, https:) and multi-token commands
            # go through the shell 'start' builtin.
            is_uri = ":" in command or command.startswith("\\")
            is_multi = " " in command.strip()
            if is_uri or is_multi:
                subprocess.Popen(f'start "" {command}', shell=True)
            else:
                # ShellExecute path: finds PATH executables AND registered
                # app names (UWP, file-type associations, etc.)
                os.startfile(command)
        else:
            subprocess.Popen(command, shell=True)
        return f"Opening {app_name}."
    except FileNotFoundError:
        # os.startfile raised because the name wasn't found anywhere;
        # fall back to the shell so the user gets a real error message.
        try:
            subprocess.Popen(f'start "" {command}', shell=True)
            return f"Opening {app_name}."
        except Exception as e:
            return f"Couldn't open {app_name}: {e}"
    except Exception as e:
        return f"Couldn't open {app_name}: {e}"


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
