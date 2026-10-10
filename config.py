"""
Jarvis configuration.

Fill in API keys as you get them. You don't need all of them —
Jarvis will only use models you've actually configured, in the
order given in MODEL_PRIORITY.
"""

import os

# --- API keys ---
# You can either hardcode them here (fine for local/personal use)
# or set them as environment variables with the same names and
# leave these as os.environ.get(...) calls.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Phase 2: voice quality upgrades (optional) ---
# Leave these blank to keep using the free local stack (faster-whisper /
# pyttsx3). Fill them in to switch to the paid, higher-quality services.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")  # a British voice ID from your ElevenLabs voice library

STT_PROVIDER = "deepgram" if DEEPGRAM_API_KEY else "local"   # "local" | "deepgram"
TTS_PROVIDER = "elevenlabs" if ELEVENLABS_API_KEY else "local"  # "local" | "elevenlabs"

# --- Phase 2: Google integrations (optional) ---
# Needs a credentials.json from Google Cloud Console (OAuth client ID,
# "Desktop app" type) placed in the jarvis/ root. See README for steps.
GOOGLE_CREDENTIALS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "credentials.json"
)
GOOGLE_TOKEN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "token.json"
)
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
]

# --- Model routing ---
# Order in which Jarvis tries models for a general query.
# Only models with a non-empty API key above will actually be tried.
MODEL_PRIORITY = ["claude", "openai", "gemini"]

# Which model to use when the user explicitly says "ask <model>"
# or "activate <model>" — maps spoken name -> internal model key.
MODEL_ALIASES = {
    "claude": "claude",
    "chatgpt": "openai",
    "gpt": "openai",
    "openai": "openai",
    "gemini": "gemini",
}

# --- Wake word ---
WAKE_WORD = "jarvis"          # openWakeWord model name to use
WAKE_WORD_THRESHOLD = 0.3      # confidence threshold, lower threshold allows "Jarvis" to trigger on first attempt

# --- Speech-to-text ---
WHISPER_MODEL_SIZE = "tiny"    # tiny/base/small/medium — tiny is fast, lightweight, and uses minimal RAM
WHISPER_DEVICE = "cpu"         # "cuda" if you want to try the GPU later

# --- Text-to-speech ---
TTS_RATE = 175                 # words per minute, roughly

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TODOS_FILE = os.path.join(DATA_DIR, "todos.json")
ALARMS_FILE = os.path.join(DATA_DIR, "alarms.json")

# --- App launcher aliases ---
# Voice-friendly name -> how to launch it on each OS.
# Add your own apps here as you need them.
APP_ALIASES = {
    "spotify": {"windows": "spotify:", "linux": "spotify"},
    "vscode": {"windows": "code", "linux": "code"},
    "vs code": {"windows": "code", "linux": "code"},
    "code": {"windows": "code", "linux": "code"},
    "visual studio code": {"windows": "code", "linux": "code"},
    "chrome": {"windows": "chrome", "linux": "google-chrome"},
    "google chrome": {"windows": "chrome", "linux": "google-chrome"},
    "browser": {"windows": "https://google.com", "linux": "xdg-open https://google.com"},
    "youtube": {"windows": "https://youtube.com", "linux": "xdg-open https://youtube.com"},
    "terminal": {"windows": "wt", "linux": "x-terminal-emulator"},
    "cmd": {"windows": "cmd", "linux": "x-terminal-emulator"},
    "command prompt": {"windows": "cmd", "linux": "x-terminal-emulator"},
    "powershell": {"windows": "powershell", "linux": "x-terminal-emulator"},
    "files": {"windows": "explorer", "linux": "xdg-open ."},
    "file explorer": {"windows": "explorer", "linux": "xdg-open ."},
    "explorer": {"windows": "explorer", "linux": "xdg-open ."},
    "notepad": {"windows": "notepad", "linux": "gedit"},
    "calculator": {"windows": "calc", "linux": "gnome-calculator"},
    "calc": {"windows": "calc", "linux": "gnome-calculator"},
    "settings": {"windows": "ms-settings:", "linux": "gnome-control-center"},
    "discord": {"windows": "discord:", "linux": "discord"},
    "whatsapp": {"windows": "whatsapp:", "linux": "whatsapp"},
    "google": {"windows": "https://google.com", "linux": "xdg-open https://google.com"},
    "edge": {"windows": "msedge", "linux": "microsoft-edge"},
    "microsoft edge": {"windows": "msedge", "linux": "microsoft-edge"},
    "camera": {"windows": "microsoft.windows.camera:", "linux": "cheese"},
    "task manager": {"windows": "taskmgr", "linux": "gnome-system-monitor"},
    "paint": {"windows": "mspaint", "linux": "drawing"},
    "word": {"windows": "winword", "linux": "libreoffice --writer"},
    "excel": {"windows": "excel", "linux": "libreoffice --calc"},
}
