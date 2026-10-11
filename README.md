# Jarvis

A voice assistant for your laptop. Simple commands (open an app, to-do list) run locally with no AI call. Everything else goes to an AI model you choose: Claude, ChatGPT, Gemini, or a free local one through Ollama. The model gets the same tools, so it can do things, not just talk.

It runs fully on your machine if you want it to. Cloud keys are optional.

## Run it (start with text, add voice after)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python doctor.py                                      # tells you what works and what's missing
python main.py --mode text --no-voice                 # type commands, no mic needed
```

Linux needs PortAudio before `pyaudio` will install: `sudo apt install portaudio19-dev`.

Once text mode works:

1. `python main.py --mode text` (adds spoken replies)
2. `python doctor.py --mic`, then set `VAD_RMS_THRESHOLD` in `.env` to what it suggests
3. `python main.py --mode ptt` (press Enter, talk, it stops when you pause)
4. `python main.py --mode wake` (always listening, say "hey jarvis")

Copy `.env.example` to `.env` for any setting. Nothing in it is required.

### A brain with no API key

Install Ollama, pull a small model that supports tools (your GPU decides how big), and put it in `.env`:

```
LOCAL_LLM_MODEL=qwen2.5:3b
```

Small local models are free and offline but worse at picking the right tool and arguments. That's useful to know for the paper too.

### Cloud keys (optional)

Put `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` in `.env` and `pip install -r requirements-cloud.txt`. Check the model names in `.env.example` against each provider's current list before you rely on them.

## What I borrowed from fullstack-agent

I looked at jaredrhod's fullstack-agent and its parts (memory vault, backtalk voice, and so on) and kept what fits a laptop assistant:

- **Memory is plain markdown, no database.** `data/identity.md` is who Jarvis is and the rules it keeps. `data/notes.md` is what you asked it to remember. Edit both by hand.
- **Free local ears and voice.** faster-whisper and pyttsx3, so no per-word billing.
- **Press to talk, stop when you pause.** Energy-based end-of-speech detection replaces the old fixed 5-second window.
- **A checker instead of an installer wizard.** `doctor.py`.

I left out the face and the hand tracking, and its dependency on Claude Code. Jarvis talks to models through their APIs.

Licensing: that repo is AGPL-3.0. I wrote this code myself from the public description (GitHub blocks automated fetching, so I couldn't read the source). If you copy files from their repo into yours, the AGPL applies to your project.

## Benchmark-ready (for the paper)

These exist so the experiments measure the real system without side effects:

- `core/router.py`: `classify_regex()` is a pure function; `route()` runs commands through a dispatch table and returns timing, path, tokens and tool calls. `force_llm=True` is the always-LLM baseline.
- `tools/mock_tools.py`: records tool calls instead of running them.
- `JARVIS_BENCHMARK=1` (the bench scripts set it): memory off, memory tool hidden, no model fallback, so every run sends the same prompt and errors get recorded.
- Models record tokens, API time, iterations and tool calls in `last_stats`.
- `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_MODEL_SIZE` are settings; `stt.load_model()`, `stt.warmup()`, `stt.reset_model()` for clean timing.
- `bench/scoring.py`: the scoring rules. Read the top of the file and fix them before you run anything.
- `bench/dryrun.py`: the Gate 2 dry run. Works with no keys (regex path only).
- `bench/sample_commands.json`: 10 sample rows in the ground-truth format. They are examples, not your dataset.
- `bench/prices.json`: fill in provider prices and the date so cost can be computed.

`python -m unittest discover -s tests` runs the tests.

Not built yet: the lightweight classifier baseline, the real dataset and audio, the STT/routing/provider experiment scripts, and the figure scripts.

## What's tested and what isn't

Tested here: the router and regex patterns, mock tools, scoring, end-of-speech detection (synthetic audio), memory, and the dry run. 13 unit tests pass.

Not tested, because the sandbox this was written in has no microphone, GPU, or API keys: the real microphone loop, the wake word model, faster-whisper on your GPU, text-to-speech, and every call to Claude, OpenAI, Gemini, Deepgram, ElevenLabs and Ollama. Expect to fix a few small things the first time you run them. The Gemini client is the least certain (Google's SDK is changing).

## Known limits

- "Opening X" is reported even if X isn't installed (the shell doesn't tell us).
- The regex router is a v1 baseline and misfires on phrases like "what is an open question". That's kept on purpose so it can be measured. Tune patterns on the dev split only.
- Timers and alarms only fire while Jarvis is running.
- On Wayland (Pop!_OS), screenshots need `gnome-screenshot`.
