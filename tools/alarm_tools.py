"""
Alarms/reminders. Phase 1 keeps this simple: an in-process timer that
fires a spoken + OS notification alert. This means alarms only fire
while Jarvis is running — persistent OS-level alarms (that survive a
reboot) are a good Phase 2 upgrade using `schedule` + autostart, or
the OS's own Task Scheduler / cron.
"""

import threading
import time

try:
    from plyer import notification
    _HAS_PLYER = True
except ImportError:
    _HAS_PLYER = False


def _fire_alarm(label: str, speak_fn):
    message = f"Alarm: {label}" if label else "Alarm"
    if _HAS_PLYER:
        try:
            notification.notify(title="Jarvis Alarm", message=message, timeout=15)
        except Exception:
            pass
    if speak_fn:
        speak_fn(message)


def set_timer(seconds: int, label: str = "", speak_fn=None) -> str:
    """Fires an alarm after `seconds` seconds. Non-blocking."""
    t = threading.Timer(seconds, _fire_alarm, args=(label, speak_fn))
    t.daemon = True
    t.start()
    minutes = seconds // 60
    if minutes >= 1:
        return f"Timer set for {minutes} minute(s)."
    return f"Timer set for {seconds} seconds."


def set_alarm_at(hour: int, minute: int, label: str = "", speak_fn=None) -> str:
    """Fires once at the next occurrence of hour:minute (24h clock)."""
    now = time.localtime()
    target = time.struct_time(
        (now.tm_year, now.tm_mon, now.tm_mday, hour, minute, 0, 0, 0, -1)
    )
    target_ts = time.mktime(target)
    now_ts = time.time()
    if target_ts <= now_ts:
        target_ts += 24 * 60 * 60  # tomorrow
    delay = target_ts - now_ts
    return set_timer(int(delay), label, speak_fn)
