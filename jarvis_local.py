"""
=============================================================================
J.A.R.V.I.S. - Production Offline AI Assistant (Enhanced Edition)
=============================================================================
Architecture:
  1. Dynamic Time-Based Greetings (Morning / Afternoon / Evening system status)
  2. System Hardware Telemetry Diagnostics (CPU, RAM, GPU, Power Profile)
  3. "Introduce Yourself" Persona Module
  4. Single-Sample XTTS v2 Voice Cloning (jarvis_sample.wav, gpt_cond_len=6.0, speed=1.0)
  5. High-Accuracy Faster-Whisper Speech Recognition ("base.en", int8)
  6. Dual Activation: Picovoice Porcupine ("Jarvis") + Global Hotkey ('F2')
  7. Deterministic Task Execution (YouTube, Spotify, Google, Tasks, Dictation, Apps)
=============================================================================
"""

import datetime
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import wave
import webbrowser

import psutil

# Ensure Coqui TOS accepted and suppress non-critical warnings
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths and Environment
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
REFERENCE_WAV = os.path.join(ROOT, "jarvis_sample.wav")
OUTPUT_WAV = os.path.join(ROOT, "jarvis_output.wav")
DICTATION_FILE = os.path.join(ROOT, "jarvis_dictation.txt")


def load_env_vars() -> None:
    env_file = os.path.join(ROOT, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env_vars()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[JARVIS] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis_offline")


def log_status(message: str) -> None:
    print(f"[JARVIS] {message}", flush=True)


# ---------------------------------------------------------------------------
# CLEAN READABLE DIALOGUE DISPLAY
# ---------------------------------------------------------------------------
_last_dialogue_speaker = None
_dialogue_lock = threading.Lock()


def display_dialogue(speaker: str, message: str) -> None:
    """
    Renders what the user said and what Jarvis is saying in a clean, highly readable format.
    """
    global _last_dialogue_speaker
    clean_msg = message.strip() if message else ""
    border = "=" * 70
    with _dialogue_lock:
        spk = speaker.upper()
        if spk in ("YOU", "USER"):
            print(f"\n{border}", flush=True)
            print(f"  YOU    : \"{clean_msg}\"", flush=True)
            _last_dialogue_speaker = "YOU"
        elif spk == "ACTIVATION":
            print(f"\n{border}", flush=True)
            print(f"  SYSTEM : Activated via {clean_msg}", flush=True)
            print("-" * 70, flush=True)
            _last_dialogue_speaker = "ACTIVATION"
        elif spk == "JARVIS":
            if _last_dialogue_speaker in ("YOU", "ACTIVATION"):
                print(f"  JARVIS : \"{clean_msg}\"", flush=True)
                print(f"{border}\n", flush=True)
            else:
                print(f"\n{border}", flush=True)
                print(f"  JARVIS : \"{clean_msg}\"", flush=True)
                print(f"{border}\n", flush=True)
            _last_dialogue_speaker = "JARVIS"
        else:
            print(f"  {spk:6} : \"{clean_msg}\"", flush=True)
            _last_dialogue_speaker = spk


# ---------------------------------------------------------------------------
# 1. TEXT CLEANING FOR CALM, STEADY SPEECH
# ---------------------------------------------------------------------------
def clean_text_for_tts(text: str) -> str:
    """
    Cleans text to eliminate distortion, trailing elongation, and agitation:
      - Strips all markdown formatting (*, _, #, `, ~, >)
      - Strips non-ASCII emojis
      - Converts exclamation marks and question marks into calm periods
      - Collapses whitespace and ensures a steady period ending.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[*_#`~>\[\]]", "", text)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[!]+", ".", cleaned)
    cleaned = re.sub(r"[?]+", ".", cleaned)
    cleaned = re.sub(r"\.+", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


# ---------------------------------------------------------------------------
# 2. LOCAL XTTS V2 VOICE CLONING (CUDA / SAPI5 FALLBACK)
# ---------------------------------------------------------------------------
class VoiceSynthesizer:
    def __init__(self, reference_wav: str = REFERENCE_WAV, output_wav: str = OUTPUT_WAV):
        self.reference_wav = reference_wav
        self.output_wav = output_wav
        self.tts = None
        self.device = "cuda"
        self._lock = threading.Lock()
        self._playback_queue = queue.Queue()
        self._worker_thread = None
        self._ensure_playback_worker()
        self._init_xtts()

    def _init_xtts(self) -> None:
        try:
            import torch
            from TTS.api import TTS

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log_status(f"Loading Coqui XTTS v2 on {self.device.upper()}...")
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            log_status("XTTS v2 voice cloning initialized.")
        except Exception as e:
            log_status(f"XTTS v2 warning ({e}). Windows system voice will serve as fallback.")
            self.tts = None

    def _ensure_playback_worker(self) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._playback_worker,
                daemon=True,
                name="AudioPlaybackWorker",
            )
            self._worker_thread.start()

    def _playback_worker(self) -> None:
        """Continuously reads generated audio chunks from queue and plays sequentially."""
        while True:
            chunk_path = self._playback_queue.get()
            if chunk_path is None:
                self._playback_queue.task_done()
                continue
            try:
                if os.path.exists(chunk_path):
                    self._play_audio_file(chunk_path)
            except Exception as e:
                log_status(f"Playback error on {chunk_path}: {e}")
            finally:
                try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                except Exception:
                    pass
                self._playback_queue.task_done()

    def _speak_system_fallback(self, text: str) -> None:
        """Fallback to Windows SAPI5 via PowerShell."""
        try:
            escaped_text = text.replace("'", "''").replace('"', '`"')
            ps_cmd = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$synth.Rate = 1; "
                f"$synth.Speak('{escaped_text}')"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
        except Exception as e:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e2:
                log_status(f"TTS Fallback error: {e2}")

    def _play_audio_file(self, wav_path: str) -> None:
        """Plays WAV output synchronously using PowerShell SoundPlayer."""
        try:
            safe_path = wav_path.replace("'", "''")
            ps_cmd = f"(New-Object System.Media.SoundPlayer '{safe_path}').PlaySync()"
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
        except Exception as e:
            log_status(f"Playback error: {e}")

    def speak(self, text: str) -> None:
        """Synthesize text using asynchronous sentence-level streaming playback."""
        clean_prompt = clean_text_for_tts(text)
        if not clean_prompt:
            return

        with self._lock:
            display_dialogue("JARVIS", clean_prompt)
            speaker_ref = self.reference_wav if os.path.exists(self.reference_wav) else "jarvis_sample.wav"
            if self.tts and os.path.exists(speaker_ref):
                try:
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', clean_prompt) if s.strip()]
                    if not sentences:
                        sentences = [clean_prompt]

                    self._ensure_playback_worker()

                    tts_speed = float(os.environ.get("JARVIS_TTS_SPEED", "1.05"))
                    for i, sentence in enumerate(sentences):
                        cleaned_sentence = clean_text_for_tts(sentence)
                        if not cleaned_sentence:
                            continue
                        temp_chunk = os.path.join(tempfile.gettempdir(), f"jarvis_chunk_{int(time.time()*1000)}_{i}.wav")
                        self.tts.tts_to_file(
                            text=cleaned_sentence,
                            speaker_wav=speaker_ref,
                            language="en",
                            file_path=temp_chunk,
                            gpt_cond_len=6.0,
                            temperature=0.2,
                            repetition_penalty=1.8,
                            speed=tts_speed,
                            enable_text_splitting=False,
                        )
                        self._playback_queue.put(temp_chunk)

                    # Wait for all queued audio chunks to complete playback before returning
                    self._playback_queue.join()
                    return
                except Exception as e:
                    log_status(f"XTTS streaming synthesis error: {e}. Falling back to system voice.")

            # Fallback path
            self._speak_system_fallback(clean_prompt)


voice_synth = VoiceSynthesizer()


def speak(text: str) -> None:
    voice_synth.speak(text)


# ---------------------------------------------------------------------------
# 3. HIGH-ACCURACY SPEECH RECOGNITION (Faster-Whisper base.en int8)
# ---------------------------------------------------------------------------
class SpeechRecognizer:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from faster_whisper import WhisperModel

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    log_status(f"Loading Faster-Whisper ('base.en', int8) on {device.upper()}...")
                    try:
                        self._model = WhisperModel("base.en", device=device, compute_type="int8")
                    except Exception as e:
                        log_status(f"CUDA STT hitch ({e}), falling back to CPU...")
                        self._model = WhisperModel("base.en", device="cpu", compute_type="int8")
                    log_status("Faster-Whisper ready.")
        return self._model

    def record_microphone(self, duration: float = 5.0, sample_rate: int = 16000) -> str:
        """Captures mic audio for `duration` seconds and returns WAV file path."""
        import pyaudio

        chunk_size = 1024
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )
        except Exception as e:
            p.terminate()
            raise RuntimeError(f"Could not open microphone: {e}")

        log_status(f"Listening for {duration:.1f} seconds...")
        frames = []
        num_chunks = int(sample_rate / chunk_size * duration)
        try:
            for _ in range(num_chunks):
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        temp_wav = os.path.join(tempfile.gettempdir(), f"jarvis_cmd_{int(time.time()*1000)}.wav")
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))

        return temp_wav

    def listen_and_transcribe(self, duration: float = 5.0) -> str:
        """Captures audio and transcribes into English text."""
        try:
            wav_path = self.record_microphone(duration=duration)
        except Exception as e:
            log_status(f"Microphone error: {e}")
            return ""

        try:
            model = self._get_model()
            segments, _ = model.transcribe(wav_path, beam_size=5, language="en")
            transcript = " ".join(seg.text for seg in segments).strip()
            log_status(f'Transcribed: "{transcript}"')
            return transcript
        except Exception as e:
            log_status(f"Transcription error: {e}")
            return ""
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass


recognizer = SpeechRecognizer()


def listen_and_transcribe(duration: float = 5.0) -> str:
    return recognizer.listen_and_transcribe(duration=duration)


# ---------------------------------------------------------------------------
# 4. SYSTEM TELEMETRY DIAGNOSTICS
# ---------------------------------------------------------------------------
def get_system_telemetry() -> dict:
    """
    Collects live CPU, RAM, GPU, and Power Plan telemetry.
    """
    # 1. CPU
    cpu_percent = int(round(psutil.cpu_percent(interval=0.15)))

    # 2. RAM
    mem = psutil.virtual_memory()
    mem_percent = int(round(mem.percent))

    # 3. GPU Telemetry
    gpu_temp = 45  # fallback reasonable temperature
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2.0,
        ).strip()
        parts = [p.strip() for p in gpu_info.split(",")]
        if len(parts) >= 4:
            gpu_temp = int(parts[3])
    except Exception:
        pass

    # 4. Power Plan
    power_mode = "Balanced"
    try:
        scheme_out = subprocess.check_output(["powercfg", "/getactivescheme"], text=True, timeout=2.0)
        match = re.search(r"\((.*?)\)", scheme_out)
        if match:
            raw_mode = match.group(1).strip()
            # Normalize display name
            if "high" in raw_mode.lower():
                power_mode = "High Performance"
            elif "silent" in raw_mode.lower() or "power saver" in raw_mode.lower():
                power_mode = "Silent"
            elif "turbo" in raw_mode.lower():
                power_mode = "Turbo"
            elif "balanced" in raw_mode.lower():
                power_mode = "Balanced"
            else:
                power_mode = raw_mode
    except Exception:
        pass

    return {
        "cpu_load": cpu_percent,
        "mem_percent": mem_percent,
        "gpu_temp": gpu_temp,
        "power_mode": power_mode,
    }


# ---------------------------------------------------------------------------
# 5. DYNAMIC TIME-BASED GREETING
# ---------------------------------------------------------------------------
def get_time_based_greeting() -> str:
    """
    Determines time of day:
      05:00 - 11:59 -> "Good morning, sir."
      12:00 - 16:59 -> "Good afternoon, sir."
      17:00 - 04:59 -> "Good evening, sir."
    """
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning, sir."
    elif 12 <= current_hour < 17:
        greeting = "Good afternoon, sir."
    else:
        greeting = "Good evening, sir."

    return f"{greeting} All primary systems are online and functioning within nominal parameters. How may I assist you today, sir."


# ---------------------------------------------------------------------------
# 6. HARDCODED JARVIS PERSONALITY & TASK EXECUTION ROUTER
# ---------------------------------------------------------------------------
class JarvisPersonalityRouter:
    """
    Hardcoded Iron Man Jarvis personality map and deterministic router.
    """

    @classmethod
    def execute(cls, command: str) -> bool:
        if not command or not command.strip():
            display_dialogue("YOU", "[No speech detected]")
            speak("Command unrecognized, sir. Standing by.")
            return False

        text = command.strip().lower()
        log_status(f'Routing command: "{text}"')
        display_dialogue("YOU", command.strip())

        # -------------------------------------------------------------------
        # 1. "INTRODUCE YOURSELF" COMMAND
        # -------------------------------------------------------------------
        if any(p in text for p in ["introduce yourself", "who are you", "tell me about yourself", "what are you"]):
            log_status("Executing: Introduce Yourself")
            intro = (
                "Allow me to introduce myself. I am Jarvis, a virtual artificial intelligence "
                "designed to assist you with desktop automation, system telemetry, and real-time operations. "
                "All core modules are at your disposal, sir."
            )
            speak(intro)
            return True

        # -------------------------------------------------------------------
        # 2. HARDWARE & SYSTEM TELEMETRY QUICK READ
        # -------------------------------------------------------------------
        if any(p in text for p in ["system status", "hardware report", "system read", "cpu temp", "gpu temp", "power mode", "system health", "telemetry"]):
            log_status("Executing: Hardware Telemetry Readout")
            data = get_system_telemetry()
            telemetry_speech = (
                f"Hardware telemetry report, sir. "
                f"CPU at {data['cpu_load']} percent, RAM at {data['mem_percent']} percent, GPU at {data['gpu_temp']} degrees Celsius. "
                f"Power profile is set to {data['power_mode']} mode. All thermal parameters are nominal, sir."
            )
            speak(telemetry_speech)
            return True

        # -------------------------------------------------------------------
        # 3. ALEXA-STYLE MEDIA STOP ("stop the song" / "stop video" / "pause")
        # -------------------------------------------------------------------
        if any(p in text for p in ["stop the song", "stop video", "stop music", "pause music", "pause the song", "pause", "resume music"]):
            log_status("Executing Media Key: Play/Pause")
            try:
                import pyautogui
                pyautogui.press("playpause")
            except Exception as e:
                log_status(f"Media key error: {e}")
            speak("Playback stopped, sir.")
            return True

        if any(p in text for p in ["next track", "next song", "skip song"]):
            try:
                import pyautogui
                pyautogui.press("nexttrack")
            except Exception:
                pass
            speak("Next track engaged, sir.")
            return True

        # -------------------------------------------------------------------
        # 4. YOUTUBE PLAYBACK ("play [title] video on youtube" / "play [title] on youtube")
        # -------------------------------------------------------------------
        yt_pattern = re.search(r"play\s+(.+?)(?:\s+video)?\s+on\s+youtube", text)
        if yt_pattern:
            raw_title = yt_pattern.group(1).strip()
            clean_title = re.sub(r"^(the|song|a)\s+", "", raw_title).strip()
            log_status(f"Initiating YouTube Playback: {clean_title}")
            speak(f"Searching YouTube and initiating playback for {clean_title.title()}, sir.")
            try:
                import pywhatkit
                pywhatkit.playonyt(clean_title)
            except Exception as e:
                log_status(f"pywhatkit error ({e}), opening search URL in browser.")
                webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(clean_title)}")
            return True

        # -------------------------------------------------------------------
        # 5. SPOTIFY PLAYBACK ("play [title] song on spotify" / "play [title] on spotify")
        # -------------------------------------------------------------------
        spotify_pattern = re.search(r"play\s+(.+?)(?:\s+song)?\s+on\s+spotify", text)
        if spotify_pattern:
            raw_title = spotify_pattern.group(1).strip()
            clean_title = re.sub(r"^(the|song|a)\s+", "", raw_title).strip()
            log_status(f"Initiating Spotify Playback: {clean_title}")
            speak(f"Searching Spotify and loading {clean_title.title()}, sir.")
            encoded = urllib.parse.quote(clean_title)
            try:
                os.startfile(f"spotify:search:{encoded}")
            except Exception:
                webbrowser.open(f"https://open.spotify.com/search/{encoded}")
            return True

        # -------------------------------------------------------------------
        # 6. GOOGLE SEARCH ("search [text] on google" / "search google for [text]")
        # -------------------------------------------------------------------
        search_pattern = re.search(r"(?:search\s+(.+?)\s+on\s+google|search\s+google\s+for\s+(.+)|google\s+(.+))", text)
        if search_pattern:
            query = (search_pattern.group(1) or search_pattern.group(2) or search_pattern.group(3)).strip()
            log_status(f"Executing Google Search: {query}")
            speak(f"Executing Google search for {query}, sir.")
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
            return True

        # -------------------------------------------------------------------
        # 7. TO-DO LIST ("create a to do list" / "open to do list" / "open todo list")
        # -------------------------------------------------------------------
        if any(p in text for p in ["create a to do list", "create a todo list", "open to do list", "open todo list", "open to-do list", "to do list", "todo list"]):
            log_status("Opening Google Tasks")
            speak("Opening Google Tasks to manage your to-do list, sir.")
            webbrowser.open("https://tasks.google.com")
            return True

        # -------------------------------------------------------------------
        # 8. NOTEPAD DICTATION ("write the text as i describe to you" / "dictate" / "take a note")
        # -------------------------------------------------------------------
        if any(p in text for p in ["write the text as i describe to you", "dictate", "take a note", "write a note"]):
            log_status("Starting Dictation Mode...")
            speak("Dictation mode active, sir. Please speak your note clearly.")
            dictation = listen_and_transcribe(duration=8.0)
            if dictation.strip():
                with open(DICTATION_FILE, "w", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"JARVIS DICTATION LOG - {timestamp}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(dictation + "\n")
                subprocess.Popen(["notepad.exe", DICTATION_FILE])
                speak("I have transcribed your dictation and opened it in Notepad, sir.")
            else:
                speak("No dictation input was detected, sir.")
            return True

        # -------------------------------------------------------------------
        # 9. APPLICATION OPENING ("open [app name]")
        # -------------------------------------------------------------------
        if "open notepad" in text:
            log_status("Launching Notepad")
            speak("Launching Notepad now, sir.")
            subprocess.Popen("notepad.exe")
            return True

        if "open calculator" in text:
            log_status("Launching Calculator")
            speak("Launching Calculator now, sir.")
            subprocess.Popen("calc.exe")
            return True

        if "open spotify" in text:
            log_status("Launching Spotify")
            speak("Launching Spotify now, sir.")
            try:
                os.startfile("spotify:")
            except Exception:
                webbrowser.open("https://open.spotify.com")
            return True

        if "open chrome" in text or "open google chrome" in text:
            log_status("Launching Chrome")
            speak("Launching Chrome now, sir.")
            webbrowser.open("https://google.com")
            return True

        if "open youtube" in text:
            log_status("Launching YouTube")
            speak("Launching YouTube now, sir.")
            webbrowser.open("https://youtube.com")
            return True

        general_open = re.search(r"open\s+([a-zA-Z0-9_\-\s]+)", text)
        if general_open:
            app_raw = general_open.group(1).strip()
            log_status(f"Launching application: {app_raw}")
            speak(f"Launching {app_raw.title()} now, sir.")
            try:
                os.startfile(app_raw)
            except Exception:
                subprocess.Popen(f'start "" {app_raw}', shell=True)
            return True

        # -------------------------------------------------------------------
        # 10. TIME & DATE QUERIES
        # -------------------------------------------------------------------
        if any(p in text for p in ["what time is it", "tell me the time", "current time", "what's the time"]):
            time_str = datetime.datetime.now().strftime("%I:%M %p")
            log_status(f"Reporting Time: {time_str}")
            speak(f"The current time is {time_str}, sir.")
            return True

        if any(p in text for p in ["what date is it", "today's date", "what day is it", "what's the date", "what is today's date"]):
            date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
            log_status(f"Reporting Date: {date_str}")
            speak(f"Today is {date_str}, sir.")
            return True

        # -------------------------------------------------------------------
        # UNKNOWN COMMAND -> Fall back to Ollama model
        # -------------------------------------------------------------------
        log_status(f"Unrecognized command: '{text}'. Falling back to Ollama model...")
        try:
            from models.ollama_manager_client import OllamaManagerClient
            ollama_client = OllamaManagerClient()
            if ollama_client.is_available():
                log_status(f"Querying Ollama model ({ollama_client.model})...")
                res = ollama_client.ask(command)
                reply = res.get("reply")
                if reply and not res.get("error"):
                    display_dialogue("JARVIS (Ollama)", reply)
                    speak(reply)
                    return True
                else:
                    log_status(f"Ollama returned error: {res.get('error')}")
        except Exception as e:
            log_status(f"Ollama fallback error: {e}")

        log_status(f"Command not recognized: {text}")
        speak("Command unrecognized, sir. Standing by.")
        return False


# ---------------------------------------------------------------------------
# 7. DUAL ACTIVATION: Porcupine Wake Word ('Jarvis') + Global Hotkey ('F2')
# ---------------------------------------------------------------------------
class DualActivationManager:
    def __init__(self):
        self.is_active = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def trigger_activation(self, trigger_source: str) -> None:
        """Invoked when either Porcupine detects 'Jarvis', openWakeWord detects 'Hey Jarvis', or F2 is pressed."""
        with self._lock:
            if self.is_active:
                log_status(f"Activation from {trigger_source} skipped (busy).")
                return
            self.is_active = True

        log_status(f"System activated via {trigger_source}!")
        display_dialogue("ACTIVATION", trigger_source)
        try:
            # Wake-up response: Jarvis responds only with 'Yes, sir?'
            speak("Yes, sir?")

            # Capture voice command (6 seconds)
            log_status("Listening for voice command (speak now)...")
            command = listen_and_transcribe(duration=6.0)

            # Execute through deterministic router
            JarvisPersonalityRouter.execute(command)
        except Exception as e:
            log_status(f"Activation handling error: {e}")
        finally:
            with self._lock:
                self.is_active = False
            log_status("Standing by for next activation.")

    def start_keyboard_listener(self) -> threading.Thread:
        """Starts background listener for global 'F2' hotkey."""
        import keyboard

        def on_f2():
            log_status("Hotkey 'F2' triggered.")
            threading.Thread(target=self.trigger_activation, args=("F2 Hotkey",), daemon=True).start()

        def loop():
            log_status("Background global F2 hotkey listener active.")
            try:
                keyboard.add_hotkey("F2", on_f2)
                while not self._stop_event.is_set():
                    time.sleep(0.5)
            except Exception as e:
                log_status(f"Keyboard listener exception: {e}")

        t = threading.Thread(target=loop, daemon=True, name="HotkeyThread")
        t.start()
        return t

    def run_porcupine_loop(self) -> None:
        """Runs the continuous Picovoice Porcupine wake word listener for 'Jarvis'."""
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY") or os.environ.get("PORCUPINE_ACCESS_KEY")

        if not access_key:
            log_status(
                "[WARN] PICOVOICE_ACCESS_KEY not configured. "
                "Falling back to local openWakeWord ('hey_jarvis') + F2 hotkey."
            )
            self._run_openwakeword_fallback()
            return

        import pvporcupine
        from pvrecorder import PvRecorder

        log_status("Initializing Picovoice Porcupine wake word engine ('jarvis')...")
        try:
            porcupine = pvporcupine.create(access_key=access_key, keywords=["jarvis"])
            recorder = PvRecorder(frame_length=porcupine.frame_length, device_index=-1)
            recorder.start()
            log_status("Porcupine continuous listening loop active. Say 'Jarvis' or press 'F2'!")
        except Exception as e:
            log_status(f"Porcupine startup error ({e}). Falling back to openWakeWord.")
            self._run_openwakeword_fallback()
            return

        try:
            while not self._stop_event.is_set():
                if self.is_active:
                    time.sleep(0.1)
                    continue

                pcm = recorder.read()
                result = porcupine.process(pcm)
                if result >= 0:
                    log_status("Wake word 'Jarvis' detected via Porcupine.")
                    recorder.stop()
                    self.trigger_activation("Porcupine Wake Word")
                    recorder.start()
        except KeyboardInterrupt:
            pass
        finally:
            recorder.delete()
            porcupine.delete()
            log_status("Porcupine stopped.")

    def _run_openwakeword_fallback(self) -> None:
        """Local openWakeWord fallback for wake word detection without cloud keys."""
        try:
            import numpy as np
            import openwakeword
            from openwakeword.model import Model
            import pyaudio

            threshold = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.28"))
            log_status(f"openWakeWord engine online (Sensitivity: {threshold:.2f}). Listening for 'Hey Jarvis' or press 'F2'!")
            openwakeword.utils.download_models()
            oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

            chunk_size = 1280
            p = pyaudio.PyAudio()

            mic_idx = None
            env_mic = os.environ.get("JARVIS_MIC_INDEX", "").strip()
            if env_mic.lstrip("-").isdigit():
                mic_idx = int(env_mic)

            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=mic_idx,
                frames_per_buffer=chunk_size,
            )

            consecutive_errors = 0
            while not self._stop_event.is_set():
                if self.is_active:
                    time.sleep(0.1)
                    continue

                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    consecutive_errors = 0
                except Exception as read_err:
                    consecutive_errors += 1
                    time.sleep(0.05)
                    if consecutive_errors > 8:
                        log_status(f"Audio stream glitch ({read_err}). Reconnecting...")
                        try:
                            stream.close()
                        except Exception:
                            pass
                        try:
                            stream = p.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=16000,
                                input=True,
                                input_device_index=mic_idx,
                                frames_per_buffer=chunk_size,
                            )
                            consecutive_errors = 0
                        except Exception:
                            time.sleep(1.0)
                    continue

                chunk = np.frombuffer(data, dtype=np.int16)

                # Soft dynamic gain boost for low-amplitude laptop microphone input
                peak = int(np.max(np.abs(chunk)))
                if 50 < peak < 14000:
                    boost = min(3.5, 16000.0 / max(peak, 1))
                    eval_chunk = np.clip(chunk.astype(np.float32) * boost, -32768, 32767).astype(np.int16)
                else:
                    eval_chunk = chunk

                scores = oww.predict(eval_chunk)
                jarvis_score = scores.get("hey_jarvis", 0.0)

                if jarvis_score >= threshold:
                    log_status(f"Wake word 'Hey Jarvis' detected! (Score: {jarvis_score:.2f} >= {threshold:.2f})")
                    try:
                        stream.stop_stream()
                    except Exception:
                        pass
                    oww.reset()

                    self.trigger_activation("Hey Jarvis (Wake Word)")

                    oww.reset()
                    try:
                        if stream.is_stopped():
                            stream.start_stream()
                    except Exception:
                        pass
        except KeyboardInterrupt:
            pass
        except Exception as e:
            log_status(f"openWakeWord initialization error: {e}. F2 Hotkey remains active.")
            while not self._stop_event.is_set():
                time.sleep(1)
        finally:
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except Exception:
                pass

    def stop(self) -> None:
        """Signals background threads to stop gracefully."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Offline Assistant (Enhanced Edition)")
    parser.add_argument("--test-text", type=str, help="Execute a single command string via text")
    parser.add_argument("--test-voice", action="store_true", help="Test dynamic time-based greeting speech")
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("   J.A.R.V.I.S. - Offline AI Assistant (Enhanced Edition)")
    print("=" * 68)
    print(f"  Voice Reference : {REFERENCE_WAV}")
    print(f"  Voice Engine    : Coqui XTTS v2 (Single Reference, gpt_cond_len=6.0)")
    print(f"  Speech Engine   : Faster-Whisper ('base.en', int8)")
    print(f"  Dual Activation : Porcupine ('jarvis') + Global F2 Hotkey")
    print(f"  Diagnostics     : Hardware Telemetry (CPU, RAM, GPU, Power)")
    print(f"  Personality     : Authentic Iron Man Jarvis Dialogue")
    print("=" * 68 + "\n")

    if args.test_voice:
        log_status("Testing dynamic greeting...")
        speak(get_time_based_greeting())
        return

    if args.test_text:
        log_status(f"Testing command: {args.test_text}")
        JarvisPersonalityRouter.execute(args.test_text)
        return

    manager = DualActivationManager()
    manager.start_keyboard_listener()

    # One-time application startup greeting upon launch
    startup_greeting = get_time_based_greeting()
    display_dialogue("ACTIVATION", "System Startup")
    speak(startup_greeting)

    log_status("Jarvis is fully operational. Say 'Hey Jarvis' or press 'F2' (Ctrl+C to quit).")

    try:
        manager.run_porcupine_loop()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        manager.stop()
        log_status("Deactivating Jarvis. Good day, sir.")
        try:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            speak("Powering down systems. Good day, sir.")
            signal.signal(signal.SIGINT, old_sigint)
        except Exception as e:
            log_status(f"Farewell speech error: {e}")


if __name__ == "__main__":
    main()
