"""
Deepgram speech-to-text. Only used when config.DEEPGRAM_API_KEY is set;
otherwise stt.py falls back to the local faster-whisper path.

Deepgram's real value is low-latency streaming transcription — this
starts with the simpler prerecorded-file approach (record a clip, send
it, get text back) since it's a drop-in replacement for the same
record_command() -> transcribe() flow Phase 1 already uses. True
mic-streaming transcription (words appearing as you speak) is a
reasonable next upgrade once this is working, since it needs an async
websocket loop instead of a single request/response call.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_client = None


def transcribe(wav_path: str) -> str:
    global _client
    if _client is None:
        from deepgram import DeepgramClient, PrerecordedOptions

        _client = DeepgramClient(config.DEEPGRAM_API_KEY)

    from deepgram import PrerecordedOptions

    with open(wav_path, "rb") as f:
        buffer_data = f.read()

    payload = {"buffer": buffer_data}
    options = PrerecordedOptions(model="nova-2", smart_format=True, language="en")

    response = _client.listen.prerecorded.v("1").transcribe_file(payload, options)
    return response.results.channels[0].alternatives[0].transcript.strip()
