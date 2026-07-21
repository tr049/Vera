"""
deepgram_speech.py  -  Deepgram STT (Nova-3) + TTS (Aura-2) backend.

Deepgram has no LLM, so this is NOT a full Provider  -  it plugs in only as a
per-stage STT/TTS override (STT_PROVIDER / TTS_PROVIDER = deepgram), composed
with a Groq/OpenAI LLM by SpeechOverrideProvider in providers.py.

Multilingual by design (matching the OpenAI/Groq path): STT always runs Nova-3
`language=multi` (EN+ES code-switching), and TTS picks a NATIVE Aura-2 voice PER
TURN from the agent's current language. Aura-2 voices are language-specific (a
persona exists only in one language  -  verified live: there is no aura-2-luna-es),
so a native-Spanish turn uses a Spanish persona and English uses an English one.

Config reuses the generic knobs  -  STT_MODEL / STT_LANGUAGE (the STT model and
language hint), and TTS_VOICE (the English voice) / TTS_VOICE_ES (the Spanish
voice, a second knob since no single Aura-2 voice covers both languages). Deepgram
has no separate TTS model  -  its voice IS the model  -  so it ignores TTS_MODEL.

Uses the official `deepgram-sdk` (>=7), lazy-imported so the base install stays
dependency-free. Batch/REST only  -  Flux (streaming/WebSocket-only) is a future pass.
"""

from __future__ import annotations

import os

from providers import _env_or_default, _pcm_to_wav

_DEFAULT_STT_MODEL = "nova-3"
_DEFAULT_STT_LANGUAGE = "multi"            # EN+ES code-switching (default)
_DEFAULT_TTS_VOICE_EN = "aura-2-luna-en"   # native English voice
_DEFAULT_TTS_VOICE_ES = "aura-2-celeste-es"  # native Spanish voice

# Preset STT model names that belong to OTHER vendors. The generic STT_MODEL/TTS_VOICE
# knobs bind to whichever backend owns the stage, so a stale value not blanked when
# switching owners (e.g. TTS_VOICE=alloy then TTS_PROVIDER=deepgram) would otherwise
# only surface as a confusing API error on the first live call -- fail fast at boot.
_NON_DEEPGRAM_STT_MODELS = frozenset({
    "whisper-large-v3-turbo", "whisper-large-v3", "whisper-1",
    "gpt-4o-mini-transcribe", "gpt-4o-transcribe",
})

# Domain terms Deepgram should bias toward (parity with OpenAI's stt_prompt).
# Verified accepted alongside language=multi.
_KEYTERMS = [
    "Vera", "Standard Queen", "Deluxe King", "Harbor Suite",
    "Family Double Queen", "Accessible Queen",
]


def _transient_errors() -> tuple[type[BaseException], ...]:
    """Transport-level failures worth one retry. httpx is lazy so the dep-free
    base install (and its tests) still work; API/auth errors are excluded on
    purpose -- retrying those only adds latency."""
    try:
        import httpx
    except ImportError:
        return (ConnectionError,)
    return (httpx.TransportError, ConnectionError)


def _retry_once(call):
    """Deepgram intermittently drops connections mid-download (observed live:
    httpx.RemoteProtocolError, 'incomplete chunked read'). One immediate retry
    on a fresh connection; no sleep -- the voice budget can't afford backoff."""
    try:
        return call()
    except _transient_errors():
        return call()


class DeepgramSpeech:
    """Deepgram speech client: transcribe / transcribe_media / synthesize."""

    name = "deepgram"

    def __init__(self):
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set DEEPGRAM_API_KEY to use Deepgram STT/TTS "
                "(STT_PROVIDER/TTS_PROVIDER=deepgram)"
            )
        from deepgram import DeepgramClient  # lazy: base install has no deepgram-sdk

        self._client = DeepgramClient(api_key=api_key)
        self.stt_model = _env_or_default("STT_MODEL", _DEFAULT_STT_MODEL)
        # "multi" = EN+ES code-switching; "en" is measurably faster but drops
        # Spanish STT -- a latency experiment, not the default.
        self.stt_language = _env_or_default("STT_LANGUAGE", _DEFAULT_STT_LANGUAGE)
        # Native voice per language (Aura-2 voices are single-language). Both are
        # generic voice knobs -- TTS_VOICE_ES is a no-op for multilingual vendors.
        self._voices = {
            "en": _env_or_default("TTS_VOICE", _DEFAULT_TTS_VOICE_EN),
            "es": _env_or_default("TTS_VOICE_ES", _DEFAULT_TTS_VOICE_ES),
        }
        if self.stt_model in _NON_DEEPGRAM_STT_MODELS:
            raise RuntimeError(
                f"STT_MODEL={self.stt_model!r} is not a Deepgram model. Blank STT_MODEL "
                "when routing STT to Deepgram (it defaults to nova-3)."
            )
        if not self._voices["en"].startswith("aura"):
            raise RuntimeError(
                f"TTS_VOICE={self._voices['en']!r} is not a Deepgram Aura voice. Blank "
                "TTS_VOICE when routing TTS to Deepgram (it defaults to aura-2-luna-en)."
            )
        # Labels for telemetry/UI (the English/default voice is the headline one).
        self.tts_model = self._voices["en"]
        self.tts_voice = self._voices["en"]

    # --- STT ---
    def _transcribe_bytes(self, audio: bytes) -> str:
        # No _retry_once here: the SDK already retries failed REQUESTS itself, and
        # wrapping it would stack into repeated full-audio uploads. The retry is
        # reserved for synthesize(), whose failure mode (connection dropped while
        # DRAINING the streamed response) the SDK's request retry cannot cover.
        resp = self._client.listen.v1.media.transcribe_file(
            request=audio,
            model=self.stt_model,
            language=self.stt_language,
            smart_format=True,
            keyterm=_KEYTERMS,
        )
        return _first_transcript(resp)

    def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw 16-bit mono PCM (CLI mic path)."""
        wav_bytes = _pcm_to_wav(pcm_int16, sample_rate).getvalue()
        return self._transcribe_bytes(wav_bytes)

    def transcribe_media(self, audio: bytes, content_type: str = "") -> str:
        """Transcribe a browser media blob (webm/ogg/mp4). Deepgram detects the
        container from the bytes, so no filename/content-type hint is needed."""
        return self._transcribe_bytes(audio)

    # --- TTS ---
    def voice_for(self, language: str | None = None) -> str:
        """The Aura-2 voice actually used for `language` (so telemetry/UI can label
        the per-turn voice instead of the static default)."""
        return self._voices.get(language or "en", self._voices["en"])

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        """Return WAV bytes, spoken by the native Aura-2 voice for `language`.
        linear16/wav so the existing WAV-only players work. Generation AND chunk
        drain sit inside the retry together -- the observed connection drops
        surface mid-download, while the iterator is being consumed."""
        return _retry_once(lambda: b"".join(self._client.speak.v1.audio.generate(
            text=text,
            model=self.voice_for(language),
            encoding="linear16",
            container="wav",
        )))


def _first_transcript(resp) -> str:
    """Pull results.channels[0].alternatives[0].transcript defensively."""
    try:
        channels = getattr(getattr(resp, "results", None), "channels", None) or []
        alternatives = getattr(channels[0], "alternatives", None) or []
        return (getattr(alternatives[0], "transcript", "") or "").strip()
    except (IndexError, AttributeError, TypeError):
        return ""
