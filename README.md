# Jarvis — Phase 1

A voice-activated assistant framework that routes commands either to
direct tool functions (open an app, manage your to-do list, set a
timer) or to an AI model of your choice (Claude, ChatGPT, Gemini) when
the request needs actual thinking.

## What Phase 1 can do

- Wake word activation ("Jarvis" — see note on training your own model below)
- Voice commands transcribed locally, no cloud STT needed
- Open apps (`"Jarvis, open Spotify"`)
- To-do list (`"add a task to buy milk"`, `"what's on my to-do list"`, `"mark buy milk as done"`)
- Timers/alarms (`"set a timer for 5 minutes"`)
- Ask any configured AI model a question, with the model able to call
  the same tools itself (`"Jarvis, ask Claude to summarize this"`)
- "What's on my screen" — takes a screenshot and asks the model about it
- Explicit model selection: `"ask gemini..."`, `"activate claude..."`

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   On Linux you'll also need PortAudio for `pyaudio` to build:
   ```bash
   sudo apt install portaudio19-dev python3-pyaudio
   ```

2. **Add at least one API key.** Edit `config.py` or set an environment
   variable:
   - Claude: https://console.anthropic.com/settings/keys
   - OpenAI: https://platform.openai.com/api-keys
   - Gemini: https://aistudio.google.com/apikey

   You don't need all three — Jarvis only uses models you've configured.

3. **Wake word model.** openWakeWord doesn't ship an exact "Jarvis"
   model out of the box. Options:
   - Train your own free custom model in ~15 minutes using their
     Colab notebook (no ML experience needed):
     https://github.com/dscripka/openWakeWord — then set `model_path`
     in `main.py`'s call to `wakeword.listen_for_wake_word(...)`.
   - Or temporarily use one of the pretrained demo words while you get
     everything else working, and swap in your trained model later.

4. **Run it:**
   ```bash
   python main.py
   ```

## Notes for your setup specifically

- You're on Pop!_OS/GNOME/Wayland day-to-day. Screenshotting (`mss`)
  doesn't work directly under Wayland's security model — the code
  falls back to `gnome-screenshot` (`sudo apt install
  gnome-screenshot`) if `mss` fails. App-opening via `subprocess`
  works fine on Wayland since it's just running a shell command.
- On your friend's Windows machine, everything works as-is — app
  opening uses `start <app>`, and screenshotting works natively via `mss`.
- `faster-whisper` on the `small` model runs fine on CPU for short
  commands even on 8GB RAM; if it feels slow, drop `WHISPER_MODEL_SIZE`
  to `"base"` in `config.py`.

## Phase 2 additions

These are all opt-in — leave the relevant API keys blank in `config.py`
and Jarvis keeps using the free local Phase 1 stack.

### Better voice quality
- **Deepgram STT**: set `DEEPGRAM_API_KEY` in `config.py`. Get a key at
  https://console.deepgram.com — free tier included. This replaces
  local Whisper for transcription (still record → send → get text back;
  true live-streaming transcription is a further upgrade from here).
- **ElevenLabs TTS**: set `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`.
  Get a key at https://elevenlabs.io — browse Voice Library for a
  British voice you like and copy its voice ID. Needs `ffplay` (comes
  with `ffmpeg`) or `mpg123` installed for playback:
  ```bash
  sudo apt install ffmpeg   # or: sudo apt install mpg123
  ```

### Google integrations (Gmail / Calendar / Drive)
1. Go to https://console.cloud.google.com, create a project.
2. Enable the **Gmail API**, **Google Calendar API**, and **Google
   Drive API** for it (APIs & Services → Library).
3. Configure the OAuth consent screen (External is fine for personal
   use; add yourself as a test user).
4. Create credentials → OAuth client ID → Application type **Desktop
   app**. Download the JSON and save it as `credentials.json` in the
   `jarvis/` root (same folder as `main.py`).
5. First time Jarvis calls a Google tool, a browser window opens for
   you to log in and grant access. After that it's saved to
   `token.json` and refreshes automatically — no repeat logins.

New voice commands this unlocks (handled by whichever LLM you're using,
via tool-calling): *"what are my recent emails"*, *"draft an email to
X about Y"*, *"what's on my calendar"*, *"schedule a meeting with X
tomorrow at 3pm"*, *"what files did I recently edit in Drive"*.

Note: `create_email_draft` creates a **draft**, not a sent email — you
still review and hit send yourself. That's deliberate; wiring it to
send automatically is a one-line change in `tools/gmail_tools.py` if
you decide you want that later.

## What's deliberately NOT in Phase 1

To keep this buildable and testable in one pass, these are left as
clear extension points rather than half-built:
- Reading/drafting real emails (needs Gmail OAuth — Phase 2)
- Real calendar scheduling (needs Google Calendar OAuth — Phase 2)
- YouTube/Spotify *control* beyond just opening the app (needs each
  service's own API/SDK — Phase 2)
- Alarms surviving a reboot (currently in-process only — Phase 2 would
  use the OS's own scheduler)
- Claude actually writing code into VS Code (this needs either the
  VS Code extension API or driving the integrated terminal — a solid
  Phase 2/3 target once the core loop above is solid)

## Project structure

```
jarvis/
  config.py              # API keys, model priority, app aliases
  main.py                 # entry point: wake word -> STT -> route -> TTS
  audio/
    wakeword.py            # openWakeWord listener
    stt.py                 # recording + faster-whisper transcription
    tts.py                 # pyttsx3 speech output
  core/
    router.py              # pattern-matches simple commands, else -> LLM
  models/
    base.py                # shared interface
    claude_client.py        # Anthropic + tool-use loop
    openai_client.py        # OpenAI + function-calling loop
    gemini_client.py        # Gemini + function-calling loop
    model_manager.py        # picks model by priority or explicit request
  tools/
    system_tools.py         # open_app, take_screenshot
    todo_tools.py           # add/list/complete to-do items
    alarm_tools.py          # timers/alarms
    tool_registry.py        # tool schemas + dispatch table
  data/
    todos.json              # local to-do storage
```
