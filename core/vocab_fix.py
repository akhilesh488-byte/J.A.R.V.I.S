"""
Conservative fuzzy vocabulary correction for STT transcripts.

Enabled only when config.VOCAB_FIX=True (JARVIS_VOCAB_FIX=1 in .env).
Default: OFF.

Strategy:
- Split the transcript into tokens.
- For each token, check if it is close (difflib ratio >= threshold) to a
  known vocabulary word from APP_ALIASES + a hardcoded command word list.
- Only correct when there is exactly ONE candidate above the threshold.
- Never correct common English words that would be over-matched.
- Log every correction so it can be reviewed.
- Does NOT touch core/router.py regex patterns.
"""

from __future__ import annotations

import difflib
import logging
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("jarvis.vocab_fix")

# Minimum similarity ratio to accept a correction.
# 0.8 is deliberately conservative to avoid over-correction.
SIMILARITY_THRESHOLD = 0.80

# Words we explicitly do NOT correct — they are common English and would
# cause false positives.
_STOPWORDS = {
    "a", "an", "the", "is", "it", "to", "do", "on", "in", "at", "be",
    "my", "me", "by", "as", "up", "of", "or", "so", "no", "yes", "ok",
    "and", "for", "but", "not", "are", "was", "has", "had", "can", "get",
}


def _build_vocab() -> list[str]:
    """Build the vocabulary from APP_ALIASES keys + command words."""
    words: set[str] = set()
    for key in config.APP_ALIASES:
        for token in key.split():
            words.add(token.lower())
    words.update([
        "open", "close", "start", "stop", "play", "pause",
        "add", "mark", "list", "show", "set", "remind",
        "todo", "timer", "note", "screenshot", "jarvis",
        "spotify", "youtube", "chrome", "firefox", "edge",
        "vscode", "notepad", "calculator", "terminal", "powershell",
        "explorer", "settings", "gmail", "maps", "netflix", "reddit",
        "github", "instagram", "twitter", "whatsapp",
        "done", "complete", "finish", "task", "grocery", "groceries",
        "call", "email", "calendar", "weather", "time", "date",
    ])
    # Remove stopwords from vocab so they don't attract corrections
    words -= _STOPWORDS
    return sorted(words)


_vocab: list[str] | None = None


def _get_vocab() -> list[str]:
    global _vocab
    if _vocab is None:
        _vocab = _build_vocab()
    return _vocab


def correct_transcript(text: str) -> str:
    """
    Apply conservative fuzzy correction to text.
    Returns corrected text; if no corrections are made, returns text unchanged.
    Only active when config.VOCAB_FIX is True.
    """
    if not config.VOCAB_FIX:
        return text

    vocab = _get_vocab()
    tokens = text.split()
    corrected = []
    for token in tokens:
        lower = token.lower().strip(r""".,!?;:'"()""")
        if lower in _STOPWORDS or len(lower) <= 2:
            corrected.append(token)
            continue
        matches = difflib.get_close_matches(lower, vocab, n=2, cutoff=SIMILARITY_THRESHOLD)
        if len(matches) == 1 and matches[0] != lower:
            logger.info("vocab_fix: '%s' -> '%s'", lower, matches[0])
            # Preserve original capitalisation style
            replacement = matches[0]
            if token[0].isupper():
                replacement = replacement.capitalize()
            corrected.append(replacement)
        else:
            corrected.append(token)

    return " ".join(corrected)
