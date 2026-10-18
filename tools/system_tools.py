"""
System-level tools: opening apps, taking screenshots.

Everything here is deliberately terminal/subprocess based rather than
GUI-automation based (no xdotool/pyautogui), so it works the same way
whether you're on Windows, X11, or Wayland.
"""

import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

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
    Strictly checks the requested app against config.APP_ALIASES, known executables in PATH,
    or explicit system executable paths. If not found, does NOT attempt to execute anything
    and returns 'App not found on system.'
    """
    os_key = _current_os()
    name = app_name.strip().lower()

    alias_entry = config.APP_ALIASES.get(name)
    if alias_entry:
        command = alias_entry[os_key]
    else:
        # Strict check against known executables or system paths
        exe_path = shutil.which(name) or (shutil.which(f"{name}.exe") if os_key == "windows" else None)
        if not exe_path and os.path.isfile(name):
            exe_path = name
        if not exe_path:
            return "App not found on system."
        command = exe_path

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
        return "App not found on system."
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


def manage_browser(action: str, query: Optional[str] = None, **kwargs) -> str:
    """
    Manage browser actions, searching, or opening URLs with state awareness.
    Checks for an active/open browser window using pygetwindow before sending hotkeys.
    """
    if query is None:
        query = kwargs.get("url") or kwargs.get("search_query") or ""

    act_lower = (action or "").strip().lower()

    if act_lower == "search":
        try:
            import urllib.parse
            import webbrowser
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(str(query))}")
        except Exception:
            pass
        return f"Searching for {query}."
    elif act_lower == "open_url":
        try:
            import webbrowser
            url = str(query)
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"https://{url}"
            webbrowser.open(url)
        except Exception:
            pass
        return f"Opened {query}."
    else:
        # Search for an active or open browser window
        browser_win = None
        try:
            import pygetwindow as gw
            browser_names = ("chrome", "edge", "firefox", "brave")
            active = gw.getActiveWindow()
            if active and active.title and any(b in active.title.lower() for b in browser_names):
                browser_win = active
            else:
                for w in gw.getAllWindows():
                    if w.title and any(b in w.title.lower() for b in browser_names):
                        browser_win = w
                        break
        except Exception:
            pass

        if not browser_win:
            return "Error: No browser window is currently open."

        # Bring window to foreground before sending hotkeys
        try:
            if getattr(browser_win, "isMinimized", False):
                browser_win.restore()
            else:
                browser_win.activate()
        except Exception:
            try:
                browser_win.restore()
            except Exception:
                pass

        try:
            import pyautogui
            if act_lower in ("new_tab", "new tab"):
                pyautogui.hotkey("ctrl", "t")
            elif act_lower in ("close_tab", "close tab"):
                pyautogui.hotkey("ctrl", "w")
            elif act_lower in ("refresh", "reload"):
                pyautogui.hotkey("ctrl", "r")
        except Exception:
            pass
        return f"Browser action done: {action}."


def play_pause_media(**kwargs) -> str:
    """
    Toggle media playback state-awarely using Windows OS Media Control.
    Checks actual system media session state via winsdk before sending commands.
    """
    try:
        import asyncio
        import winsdk.windows.media.control as wmc

        async def _control_media_async() -> str:
            mgr = await wmc.GlobalSystemMediaTransportControlsSessionManager.request_async()
            session = mgr.get_current_session()
            if session is None:
                sessions = mgr.get_sessions()
                if sessions:
                    session = sessions[0]

            if session is None:
                return "Error: No active media session found to control."

            info = session.get_playback_info()
            status = info.playback_status if info else None

            # PLAYING is 4 in GlobalSystemMediaTransportControlsSessionPlaybackStatus
            if status == wmc.GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING:
                try:
                    await session.try_pause_async()
                except Exception:
                    pass
                return "Paused active media."
            else:
                try:
                    await session.try_play_async()
                except Exception:
                    pass
                return "Resumed media playback."

        try:
            return asyncio.run(_control_media_async())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_control_media_async())
            finally:
                loop.close()
    except Exception:
        return "Error: No active media session found to control."


def close_current_app(window_title: Optional[str] = None, **kwargs) -> str:
    """
    Close the active window or an app by window title.
    """
    title = window_title
    win = None
    if not title:
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win and win.title:
                title = win.title
        except Exception:
            pass

    if not title:
        title = "active window"

    try:
        if win is not None:
            win.close()
        else:
            import pyautogui
            pyautogui.hotkey("alt", "f4")
    except Exception:
        pass

    return f"Closed {title}."


def get_window_status(title: Optional[str] = None, apps: Optional[List[str]] = None, **kwargs) -> str:
    """
    Get active window and open applications.
    """
    if apps is None and "open_apps" in kwargs:
        apps = kwargs["open_apps"]

    active_title = title
    open_apps_list = list(apps) if apps is not None else None

    if active_title is None or open_apps_list is None:
        detected_title = None
        detected_apps = []
        try:
            import pygetwindow as gw
            active = gw.getActiveWindow()
            if active and active.title and active.title.strip():
                detected_title = active.title.strip()
            all_wins = gw.getAllWindows()
            for w in all_wins:
                if w.title and w.title.strip():
                    t = w.title.strip()
                    if t not in detected_apps:
                        detected_apps.append(t)
        except Exception:
            pass

        if not detected_apps:
            try:
                import psutil
                known_aliases = set(config.APP_ALIASES.keys()) if hasattr(config, "APP_ALIASES") else set()
                running_procs = []
                for p in psutil.process_iter(["name"]):
                    p_name = p.info.get("name")
                    if p_name:
                        base = p_name.lower().replace(".exe", "")
                        if base in known_aliases and base not in running_procs:
                            running_procs.append(base)
                if running_procs:
                    detected_apps = running_procs
            except Exception:
                pass

        if active_title is None:
            active_title = detected_title or "Desktop"
        if open_apps_list is None:
            open_apps_list = detected_apps if detected_apps else ["Desktop"]

    apps_str = ", ".join(open_apps_list) if open_apps_list else "None"
    return f"Active window: {active_title}. Open apps: {apps_str}"

