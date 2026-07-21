"""Streaming-cascade helpers (Option A): sentence buffering + the on/off gate.

Kept dependency-free and provider-agnostic so the CLI turn loop and (later) the
browser server share the same sentence segmentation and the same rule for when
streaming is active. Streaming is OFF unless `STREAMING` is truthy AND the
provider actually supports it, so the offline mock / eval path is never affected.
"""

from __future__ import annotations

import os

import re

_TRUTHY = {"1", "true", "on", "yes"}

# A sentence ends on . ! or ? (with any run of them, plus an optional closing
# quote/bracket), followed by whitespace. Requiring the trailing space avoids
# splitting decimals ("$3.50") or mid-number; the final sentence of a reply has
# no trailing space and is emitted by flush() instead.
_SENTENCE_END = re.compile(r"""[.!?]+["')\]]*\s""")


class SentenceBuffer:
    """Accumulate streamed text deltas and emit complete sentences.

    `feed(delta)` returns the sentences completed by this delta (usually empty
    until a terminator arrives); `flush()` returns whatever partial text is left
    at end of stream. Splitting on sentence boundaries keeps each TTS chunk
    natural and lets playback start on the first sentence.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._buf += delta
        out: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                break
            sentence = self._buf[: match.end()].strip()
            self._buf = self._buf[match.end():]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str:
        tail = self._buf.strip()
        self._buf = ""
        return tail


def streaming_enabled(provider, *, text_mode: bool = False) -> bool:
    """True when low-latency streaming should run for this turn.

    Gated by BOTH config (`STREAMING` truthy, default off) AND capability: the
    provider must expose `chat_stream` and use cloud TTS, and we must not be in
    text-only mode. So `STREAMING=on` with the mock (no `chat_stream`) or with
    `TTS_BACKEND=system` stays fully batch, and the offline / eval path is
    untouched. Callers pass an `on_delta` callback only when this returns True.
    """
    if os.getenv("STREAMING", "off").strip().lower() not in _TRUTHY:
        return False
    if text_mode:
        return False
    if getattr(provider, "chat_stream", None) is None:
        return False
    return getattr(provider, "tts_backend", "provider") == "provider"
