"""
Jarvis configuration.

Everything can be set with environment variables or a .env file in this
folder (copy .env.example to .env). Nothing here needs editing for a
fully local, no-keys run: text mode + local Whisper + local voice works
out of the box, and a local LLM (Ollama) is optional.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_env(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(os.path.join(ROOT, ".env"))


def _bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _float_or_none(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    if v.strip().lower() in ("", "none", "null"):
        return None
    return float(v)


# --- Cloud API keys (all optional) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")

# --- Models (check these against each provider's current model list) ---
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Local LLM through Ollama's OpenAI-compatible endpoint (free, offline) ---
# Example: ollama pull qwen2.5:3b   then   LOCAL_LLM_MODEL=qwen2.5:3b
# Pick a model that supports tool calling; a 4 GB GPU means small models.
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")

# Order Jarvis tries models in. Override: JARVIS_MODEL_PRIORITY=local,claude
MODEL_PRIORITY = [
    m.strip()
    for m in os.environ.get("JARVIS_MODEL_PRIORITY", "claude,openai,gemini,local").split(",")
    if m.strip()
]
MODEL_ALIASES = {
    "claude": "claude",
    "chatgpt": "openai",
    "gpt": "openai",
    "openai": "openai",
    "gemini": "gemini",
    "local": "local",
    "ollama": "local",
}

# --- LLM behaviour ---
LLM_TEMPERATURE = _float_or_none("JARVIS_TEMPERATURE", 0.3)  # "none" = don't send it
LLM_MAX_TOKENS = int(os.environ.get("JARVIS_MAX_TOKENS", "1024"))
MAX_TOOL_ITERATIONS = 5

# --- Input mode: wake | ptt | text ---
INPUT_MODE = os.environ.get("JARVIS_MODE", "wake")

# --- Wake word (openWakeWord ships a pretrained "hey_jarvis" model; verify
# with doctor.py. A custom .onnx path also works here.) ---
WAKE_WORD_MODEL = os.environ.get("WAKE_WORD_MODEL", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.28"))

# --- Recording / end-of-speech detection (energy based, int16 scale) ---
VAD_RMS_THRESHOLD = float(os.environ.get("VAD_RMS_THRESHOLD", "500"))
VAD_SILENCE_SECONDS = float(os.environ.get("VAD_SILENCE_SECONDS", "1.0"))
VAD_MIN_SECONDS = 0.6
VAD_NO_SPEECH_TIMEOUT = 5.0
VAD_MAX_SECONDS = 15.0

# --- Speech-to-text ---
_has_cuda = False
try:
    import torch
    _has_cuda = torch.cuda.is_available()
except Exception:
    pass

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")  # tiny | small | small.en | medium.en | large-v3-turbo | distil-large-v3
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda" if _has_cuda else "cpu")  # cpu | cuda | auto
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16" if WHISPER_DEVICE == "cuda" else "int8")  # float16 | int8 | int8_float16 (CUDA)
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")  # "en" | "" for auto-detect
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
WHISPER_VAD_FILTER = _bool("WHISPER_VAD_FILTER", True)   # filters non-speech before decoding

# STT engine selection (faster_whisper | deepgram | openai | elevenlabs_scribe | cohere_transcribe)
# legacy: if DEEPGRAM_API_KEY is set and STT_ENGINE is not overridden, use deepgram
STT_ENGINE = os.environ.get("STT_ENGINE", "deepgram" if DEEPGRAM_API_KEY else "faster_whisper")
# Comma-separated fallback chain tried when the primary engine fails or is unavailable
STT_FALLBACK = [e.strip() for e in os.environ.get("STT_FALLBACK", "faster_whisper").split(",") if e.strip()]
# keep the old name working for backwards-compat
STT_PROVIDER = "deepgram" if DEEPGRAM_API_KEY else "local"

# --- Microphone ---
MIC_INDEX = None  # None = system default; set JARVIS_MIC_INDEX=<int> to pick a device
_mic_idx = os.environ.get("JARVIS_MIC_INDEX", "")
if _mic_idx.strip().lstrip("-").isdigit():
    MIC_INDEX = int(_mic_idx)

# --- Audio capture quality ---
# 300 ms of audio captured before voice activity so the first syllable isn't cut
AUDIO_PREROLL_MS = int(os.environ.get("AUDIO_PREROLL_MS", "300"))
# 200 ms of silence appended after end-of-speech for trailing-syllable safety
AUDIO_TRAILING_PAD_MS = int(os.environ.get("AUDIO_TRAILING_PAD_MS", "200"))
# Peak-normalise quiet recordings before transcription (safe; only raises quiet audio)
AUDIO_NORMALIZE = _bool("AUDIO_NORMALIZE", True)
# Seconds to wait after TTS finishes before opening the mic (avoids speaker echo)
TTS_POST_SPEAK_DELAY = float(os.environ.get("TTS_POST_SPEAK_DELAY", "0.3"))

# --- Paths ---
DATA_DIR = os.path.join(ROOT, "data")
TODOS_FILE = os.path.join(DATA_DIR, "todos.json")
ALARMS_FILE = os.path.join(DATA_DIR, "alarms.json")
IDENTITY_FILE = os.path.join(DATA_DIR, "identity.md")
NOTES_FILE = os.path.join(DATA_DIR, "notes.md")

# --- Debug audio: save every recording for inspection ---
DEBUG_AUDIO = _bool("JARVIS_DEBUG_AUDIO", False)
DEBUG_AUDIO_DIR = os.path.join(DATA_DIR, "recordings")

# --- Vocabulary fuzzy correction (Task 6, default OFF) ---
VOCAB_FIX = _bool("JARVIS_VOCAB_FIX", False)

# --- Text-to-speech ---
TTS_RATE = int(os.environ.get("TTS_RATE", "175"))
TTS_PROVIDER = "elevenlabs" if ELEVENLABS_API_KEY else "local"

# --- Memory (plain markdown, no database) ---
USE_MEMORY = _bool("JARVIS_USE_MEMORY", True)

# --- Benchmark mode: turns off memory/extra tools and fallbacks so runs are
# comparable. The bench scripts set this themselves. ---
BENCHMARK_MODE = _bool("JARVIS_BENCHMARK", False)
# Timer regex is OFF by default on purpose: decide on the dev set before
# freezing the router (see blueprint, Phase 2.3).
ENABLE_TIMER_REGEX = _bool("JARVIS_TIMER_REGEX", False)

# --- Google (optional) ---
GOOGLE_CREDENTIALS_FILE = os.path.join(ROOT, "credentials.json")
GOOGLE_TOKEN_FILE = os.path.join(ROOT, "token.json")
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
]

# --- App launcher aliases (no leading "start": open_app adds it on Windows) ---
# Windows values:
#   - Single word  -> os.startfile() / ShellExecute (searches PATH + registry)
#   - URI scheme   -> subprocess + shell 'start' (e.g. shell: and https: URIs)
#   - Multi-word   -> subprocess + shell 'start'
APP_ALIASES = {
    # --- Code editors ---
    "vscode": {"windows": "code", "linux": "code"},
    "visual studio code": {"windows": "code", "linux": "code"},
    # --- Music / media ---
    "spotify": {"windows": "spotify:", "linux": "spotify"},
    # --- Browsers ---
    "browser": {"windows": "https://google.com", "linux": "xdg-open https://google.com"},
    "chrome": {"windows": "https://google.com", "linux": "google-chrome"},
    "google chrome": {"windows": "https://google.com", "linux": "google-chrome"},
    "firefox": {"windows": "firefox", "linux": "firefox"},
    "edge": {"windows": "microsoft-edge:", "linux": "microsoft-edge"},
    # --- Popular websites & apps (open in default browser or desktop protocol) ---
    "youtube": {"windows": "https://youtube.com", "linux": "xdg-open https://youtube.com"},
    "you tube": {"windows": "https://youtube.com", "linux": "xdg-open https://youtube.com"},
    "u tube": {"windows": "https://youtube.com", "linux": "xdg-open https://youtube.com"},
    "google": {"windows": "https://google.com", "linux": "xdg-open https://google.com"},
    "gmail": {"windows": "https://mail.google.com", "linux": "xdg-open https://mail.google.com"},
    "google maps": {"windows": "https://maps.google.com", "linux": "xdg-open https://maps.google.com"},
    "maps": {"windows": "https://maps.google.com", "linux": "xdg-open https://maps.google.com"},
    "whatsapp": {"windows": "whatsapp:", "linux": "xdg-open https://web.whatsapp.com"},
    "whats app": {"windows": "whatsapp:", "linux": "xdg-open https://web.whatsapp.com"},
    "instagram": {"windows": "https://instagram.com", "linux": "xdg-open https://instagram.com"},
    "twitter": {"windows": "https://twitter.com", "linux": "xdg-open https://twitter.com"},
    "x": {"windows": "https://x.com", "linux": "xdg-open https://x.com"},
    "reddit": {"windows": "https://reddit.com", "linux": "xdg-open https://reddit.com"},
    "github": {"windows": "https://github.com", "linux": "xdg-open https://github.com"},
    "netflix": {"windows": "https://netflix.com", "linux": "xdg-open https://netflix.com"},
    "chatgpt": {"windows": "https://chat.openai.com", "linux": "xdg-open https://chat.openai.com"},
    # --- System apps ---
    "terminal": {"windows": "cmd", "linux": "x-terminal-emulator"},
    "powershell": {"windows": "powershell", "linux": "bash"},
    "files": {"windows": "explorer", "linux": "xdg-open ."},
    "file explorer": {"windows": "explorer", "linux": "xdg-open ."},
    "notepad": {"windows": "notepad", "linux": "gedit"},
    "calculator": {"windows": "calc", "linux": "gnome-calculator"},
    "task manager": {"windows": "taskmgr", "linux": "gnome-system-monitor"},
    "settings": {"windows": "ms-settings:", "linux": "gnome-control-center"},
    "control panel": {"windows": "control", "linux": "gnome-control-center"},
}
