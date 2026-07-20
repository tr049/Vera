"""
providers.py  -  one adaptor, two backends: Groq and OpenAI.

Groq speaks the OpenAI API dialect, so a single code path covers both  -  only
base_url, api_key, and model names differ. Switch with PROVIDER=groq|openai in
.env; move to your OpenAI key later by flipping that one value.

Exposes three stages the voice loop needs:
    chat(messages, tools)        -> LLM turn (OpenAI-style tool calling)
    transcribe(pcm_int16, rate)  -> STT (Whisper)
    synthesize(text)             -> TTS; returns WAV bytes, or None if it
                                    already played via the system voice command
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import unicodedata
import wave
from types import SimpleNamespace as NS

# Sensible defaults per backend. Any of these can be overridden in .env.
PRESETS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # 70b = reliable tool-calling; swap to llama-3.1-8b-instant for lower latency.
        "llm_model": "llama-3.3-70b-versatile",
        "stt_model": "whisper-large-v3-turbo",
        "tts_model": "canopylabs/orpheus-v1-english",
        "tts_voice": "troy",
    },
    "openai": {
        "base_url": None,                # SDK default endpoint
        "api_key_env": "OPENAI_API_KEY",
        # Non-reasoning, low-latency, native function calling on /v1/chat/completions
        # — the right fit for this tool-driven voice agent. (gpt-5.6-* reasoning models
        # reject function tools + reasoning_effort here; they need /v1/responses.)
        "llm_model": "gpt-4.1-mini",
        "stt_model": "gpt-4o-mini-transcribe",
        "tts_model": "gpt-4o-mini-tts",
        "tts_voice": "alloy",
    },
}

DEFAULT_STT_PROMPT = (
    "Vera Hotel reservations conversation in English or Spanish. "
    "Hotel vocabulary: reservation, booking, check-in, check-out, cancellation policy, "
    "pet policy, parking, breakfast, accessibility, habitación, reserva, política de "
    "cancelación, mascotas, estacionamiento, desayuno, accesibilidad."
)


def _env_or_default(key: str, default: str) -> str:
    """Return a non-empty environment override or the provider preset.

    A copied .env template can leave a comment after an empty assignment.
    Some dotenv versions preserve that comment as the value, which would send
    an invalid model ID to the provider.
    """
    value = os.getenv(key, "").strip()
    if not value or value.startswith("#"):
        return default
    return value


# GPT-5 family and o-series are reasoning models: they reject `temperature` and
# steer via `reasoning_effort` instead. Prefix-match routes Provider.chat() below.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class Provider:
    """Configured client for one backend. Read from .env on construction."""

    def __init__(self, name: str | None = None):
        name = (name or os.getenv("PROVIDER", "groq")).lower()
        if name not in PRESETS:
            raise ValueError(f"Unknown PROVIDER {name!r}; use one of {list(PRESETS)}")
        self.name = name
        p = PRESETS[name]

        api_key = os.getenv(p["api_key_env"])
        if not api_key:
            raise RuntimeError(f"Set {p['api_key_env']} in your .env (PROVIDER={name})")
        from openai import OpenAI  # lazy: the mock path needs no SDK installed
        self.client = OpenAI(api_key=api_key, base_url=p["base_url"])

        # Per-stage overrides fall back to the preset.
        self.llm_model = _env_or_default("LLM_MODEL", p["llm_model"])
        self.stt_model = _env_or_default("STT_MODEL", p["stt_model"])
        self.stt_prompt = _env_or_default("STT_PROMPT", DEFAULT_STT_PROMPT)
        self.tts_model = _env_or_default("TTS_MODEL", p["tts_model"])
        self.tts_voice = _env_or_default("TTS_VOICE", p["tts_voice"])
        self.tts_instructions = os.getenv("TTS_INSTRUCTIONS")
        # "provider" = cloud TTS; "system" = local system voice command.
        self.tts_backend = os.getenv("TTS_BACKEND", "provider").lower()

    # --- LLM ---
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice=None,
    ):
        """One chat-completion call. Returns the raw SDK response."""
        kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": (tool_choice or "auto") if tools else None,
        }
        # Escape hatch if you override LLM_MODEL to a reasoning model (gpt-5*, o-series):
        # they reject a custom `temperature`, and on /v1/chat/completions they reject
        # function tools unless reasoning is OFF — so reasoning_effort defaults to 'none'
        # (also the lowest latency). Non-reasoning models (gpt-4.1/4o, Groq llama) keep 0.3.
        if self.llm_model.startswith(_REASONING_MODEL_PREFIXES):
            kwargs["reasoning_effort"] = os.getenv("REASONING_EFFORT", "none")
        else:
            kwargs["temperature"] = 0.3
        return self.client.chat.completions.create(**kwargs)

    # --- STT ---
    def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw 16-bit mono PCM via Whisper."""
        wav = _pcm_to_wav(pcm_int16, sample_rate)
        wav.name = "turn.wav"  # SDK infers format from the filename
        transcription_args = {
            "model": self.stt_model,
            "file": wav,
            "response_format": "text",
        }
        if self.stt_prompt:
            transcription_args["prompt"] = self.stt_prompt
        resp = self.client.audio.transcriptions.create(
            **transcription_args,
        )
        return (resp if isinstance(resp, str) else resp.text).strip()

    def transcribe_media(self, audio: bytes, content_type: str = "") -> str:
        """Transcribe a browser media blob (webm/ogg/mp4). Unlike `transcribe`
        (raw PCM), the browser sends a container, so tag the buffer by mimetype
        (the SDK infers format from the filename)."""
        audio_file = io.BytesIO(audio)
        if "mp4" in content_type:
            audio_file.name = "caller.mp4"
        elif "ogg" in content_type:
            audio_file.name = "caller.ogg"
        else:
            audio_file.name = "caller.webm"
        transcription_args = {
            "model": self.stt_model,
            "file": audio_file,
            "response_format": "text",
        }
        if self.stt_prompt:
            transcription_args["prompt"] = self.stt_prompt
        resp = self.client.audio.transcriptions.create(**transcription_args)
        return (resp if isinstance(resp, str) else resp.text).strip()

    # --- TTS ---
    def synthesize(self, text: str, language: str | None = None) -> bytes | None:
        """Return WAV bytes for `text`, or None if played directly by the OS.
        `language` is accepted for interface parity but unused  -  OpenAI/Groq
        voices are multilingual; only Deepgram selects a per-language voice."""
        if self.tts_backend == "system":
            subprocess.run([os.getenv("SYSTEM_TTS_CMD", "say"), text], check=False)
            return None
        speech_args = {
            "model": self.tts_model,
            "voice": self.tts_voice,
            "input": text,
            "response_format": "wav",
        }
        if self.tts_instructions:
            speech_args["instructions"] = self.tts_instructions
        resp = self.client.audio.speech.create(
            **speech_args,
        )
        return resp.content


# --- audio helpers ---

def _pcm_to_wav(pcm_int16: bytes, sample_rate: int) -> io.BytesIO:
    """Wrap raw 16-bit mono PCM samples into an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm_int16)
    buf.seek(0)
    return buf


def is_noise_transcript(text: str) -> bool:
    """True when a transcript carries no speech (empty, or punctuation/noise only).

    STT models hallucinate stray tokens on silence or background noise. This is
    deliberately conservative: it rejects only transcripts with no alphanumeric
    content, so a real one-word answer like "Yes" / "Si" / "No" always passes.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    return not any(character.isalnum() for character in stripped)


# Whole-transcript phrases that are almost never a real hotel caller turn: generic
# AI-assistant greetings and content-farm sign-offs that STT emits on silence, or
# transcribes from the agent's OWN playback bleeding into the mic (common on a phone
# speaker). Matched against the full normalized transcript so a caller who merely
# uses one of these words is never dropped.
_STT_HALLUCINATIONS = frozenset({
    "how can i help you",
    "how can i help you today",
    "how may i help you",
    "how can i assist you",
    "how can i assist you today",
    "como puedo ayudarte",
    "como puedo ayudarte hoy",
    "como puedo ayudarle",
    "como puedo asistirte",
    "en que puedo ayudarte",
    "en que puedo ayudarle",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
})


def _normalize_transcript(text: str) -> str:
    """Lowercase, strip accents (so "cómo" == "como"), keep only words/digits."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", no_accents))


def is_stt_hallucination(text: str) -> bool:
    """True for transcripts that are an STT hallucination or agent-playback echo
    rather than a real caller turn (e.g. "ChatGPT, ¿Cómo puedo ayudarte hoy?").

    Only whole-transcript matches count, plus any mention of "chatgpt"/"openai"
    (a hotel caller never says those), so real speech is not affected.
    """
    normalized = _normalize_transcript(text)
    if not normalized:
        return False
    if "chatgpt" in normalized or "openai" in normalized:
        return True
    return normalized in _STT_HALLUCINATIONS


# --- Mock backend: full offline end-to-end, no network / key / SDK ---

class MockProvider:
    """Drop-in stand-in for Provider. Rule-based LLM, scripted STT, no-op TTS.

    Same interface (chat / transcribe / synthesize) so voice_loop.py and
    agent.py can't tell the difference. Use for demos, CI, and testing the
    loop without touching Groq/OpenAI. Enable with PROVIDER=mock.
    """

    name = "mock"

    def __init__(self):
        self.llm_model = "mock-llm"
        self.stt_model = "mock-stt"
        self.tts_model = "mock-tts"
        self.tts_voice = "mock"
        self.tts_backend = os.getenv("TTS_BACKEND", "print").lower()
        # Scripted transcripts for mic mode (there's no offline STT); cycles.
        self._stt_script = [
            "I need a room from August 12 to August 14 for two guests.",
            "Book it for Priya Shah, priya@example.com.",
            "Can I speak to a person?",
            "Goodbye",
        ]
        self._stt_i = 0

    def chat(self, messages: list[dict], tools=None, tool_choice=None):
        """Rule-based reply mimicking OpenAI-style tool calling."""
        last = messages[-1]
        spanish = "Current response language: Spanish" in messages[0].get("content", "")
        forced_tool = _tool_choice_name(tool_choice)
        if forced_tool == "search_hotel_knowledge" and last.get("role") == "user":
            return _mk_tool(forced_tool, {"query": last.get("content") or ""})
        # After a tool ran, speak a reply built from its result.
        if last.get("role") == "tool":
            result = last["content"]
            if result.lower().startswith("response language set to spanish"):
                original = _last_user_text(messages).lower()
                if _mock_knowledge_request(original):
                    return _mk_tool("search_hotel_knowledge", {"query": original})
                if _mock_off_topic(original):
                    return _mk_text("Solo puedo ayudar con reservas de hotel. ¿Quiere reservar, cambiar o cancelar una estancia?")
                return _mk_text("Claro. Puedo ayudarle con una reserva en Vera Hotel.")
            if result.lower().startswith("response language set to english"):
                original = _last_user_text(messages).lower()
                if _mock_knowledge_request(original):
                    return _mk_tool("search_hotel_knowledge", {"query": original})
                if _mock_off_topic(original):
                    return _mk_text("I can only help with hotel reservations. Are you looking to book, change, or cancel a stay?")
                return _mk_text("Of course. I can continue in English with your Vera Hotel reservation.")
            if result.lower().startswith("available rooms"):
                if spanish:
                    return _mk_text(f"{result} ¿Quiere que reserve una de estas habitaciones?")
                return _mk_text(f"{result} Would you like me to book one of these?")
            if result.lower().startswith("no matching rooms"):
                if spanish:
                    return _mk_text(
                        "No tengo una habitación disponible para esas fechas. "
                        "¿Quiere que le comunique con la recepción?"
                    )
                return _mk_text(
                    "I don't have a room available for those dates. "
                    "I can connect you with the front desk if you'd like."
                )
            if result.lower().startswith("booking confirmed"):
                if spanish:
                    confirmation = re.search(r"VH-\d+", result)
                    code = confirmation.group(0) if confirmation else "confirmada"
                    return _mk_text(f"La reserva está confirmada. Su número de confirmación es {code}.")
                return _mk_text(result)
            if result.lower().startswith("grounded hotel knowledge"):
                tool_args = _previous_tool_arguments(messages)
                return _mk_text(_grounded_policy_reply(result, spanish, tool_args.get("query", "")))
            if result.lower().startswith("transferring") and spanish:
                return _mk_text("Le transfiero a la recepción.")
            if result.lower().startswith("ending") and spanish:
                return _mk_text("Gracias por llamar a Vera Hotel. Adiós.")
            return _mk_text(result)  # transfer / hangup / not-found: speak as-is

        text = (last.get("content") or "").lower()
        tokens = set(re.findall(r"[\wáéíóúüñ]+", text, flags=re.UNICODE))
        if any(phrase in text for phrase in (
            "speak spanish", "switch to spanish", "spanish please", "habla español",
            "hable español", "en español",
        )):
            return _mk_tool("set_language", {"language": "es"})
        if any(phrase in text for phrase in (
            "speak english", "switch to english", "switch back to english",
            "back to english", "return to english", "english please", "english again",
            "habla inglés", "hable inglés", "en inglés", "habla ingles",
        )):
            return _mk_tool("set_language", {"language": "en"})
        if _mock_knowledge_request(text):
            return _mk_tool("search_hotel_knowledge", {"query": last.get("content") or ""})
        if any(w in text for w in ("bye", "goodbye", "that's all", "thats all",
                                   "nothing else", "no thanks", "hang up", "adiós", "adios")):
            return _mk_tool("end_call", {})
        if _mock_off_topic(text):
            if spanish:
                return _mk_text("Solo puedo ayudar con reservas de hotel. ¿Quiere reservar, cambiar o cancelar una estancia?")
            return _mk_text("I can only help with hotel reservations. Are you looking to book, change, or cancel a stay?")
        if any(phrase in text for phrase in (
            "another reservation", "another guest", "other guest", "someone else's",
        )):
            if spanish:
                return _mk_text("No puedo revelar datos de otro huésped. Solo puedo ayudar con su propia reserva de hotel.")
            return _mk_text("I cannot disclose another guest's information. I can only help with your own hotel reservation.")
        if tokens & {"human", "person", "representative", "agent", "operator", "persona", "recepción"}:
            return _mk_tool("transfer_to_human", {})
        if any(w in text for w in ("change", "cancel", "modify", "front desk")):
            return _mk_tool("transfer_to_human", {})
        if any(w in text for w in ("book", "reserve", "yes", "confirm", "reservar", "confirmo")) and any(
            w in text for w in ("name", "email", "@", "phone", "priya", "shah", "nombre")
        ):
            return _mk_tool("create_booking", {
                "check_in": "August 12",
                "check_out": "August 14",
                "guests": 2,
                "room_type": "standard",
                "guest_name": "Priya Shah",
                "contact": "priya@example.com",
            })
        if any(w in text for w in (
            "room", "hotel", "stay", "book", "reservation", "guests", "guest",
            "habitación", "habitacion", "reserva", "personas", "huéspedes", "huespedes",
        )):
            return _mk_tool("check_availability", {
                "check_in": "August 12",
                "check_out": "August 14",
                "guests": _mock_guest_count(text),
            })
        if spanish:
            return _mk_text("Solo puedo ayudar con reservas de hotel. ¿Quiere reservar, cambiar o cancelar una estancia?")
        return _mk_text("I can help with hotel reservations only. Would you like to book, change, or cancel a stay?")

    def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        """No offline STT  -  return the next scripted phrase (demo mode)."""
        phrase = self._stt_script[self._stt_i % len(self._stt_script)]
        self._stt_i += 1
        return phrase

    def transcribe_media(self, audio: bytes = b"", content_type: str = "") -> str:
        """Browser-audio path in mock mode: same scripted phrases as transcribe."""
        return self.transcribe(b"")

    def synthesize(self, text: str, language: str | None = None) -> bytes | None:
        """No cloud TTS. Optionally use a local voice command; else print-only."""
        if self.tts_backend == "system":
            subprocess.run([os.getenv("SYSTEM_TTS_CMD", "say"), text], check=False)
        return None  # voice_loop already prints the agent's text


def _mk_text(content: str):
    return NS(choices=[NS(message=NS(content=content, tool_calls=None))])


def _mk_tool(name: str, args: dict):
    tc = NS(id=f"call_{name}", type="function",
            function=NS(name=name, arguments=json.dumps(args)))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))])


def _tool_choice_name(tool_choice) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function") or {}
    return function.get("name")


def _last_user_text(messages: list[dict]) -> str:
    return next(
        (message.get("content") or "" for message in reversed(messages) if message.get("role") == "user"),
        "",
    )


def _mock_knowledge_request(text: str) -> bool:
    return any(word in text for word in (
        "cancellation policy", "cancel policy", "check-in", "check in", "check-out",
        "check out", "parking", "pets", "pet policy", "breakfast", "accessible",
        "accessibility", "policy", "estacionamiento", "mascotas", "desayuno",
    ))


def _mock_off_topic(text: str) -> bool:
    return any(word in text for word in (
        "weather", "news", "sports", "stock", "joke", "trivia", "clima", "noticias",
    ))


def _mock_guest_count(text: str) -> int:
    """Parse a digit guest count from the caller text (e.g. "9 guests" -> 9).

    Defaults to 2 (the demo party size) when no explicit number is present, so
    existing word-based flows ("two guests", "dos personas") are unchanged. This
    lets the offline mock actually reach the no-availability path for a large
    party instead of always fitting a room.
    """
    match = re.search(
        r"(\d+)\s*(?:guests?|people|persons?|adults?|personas?|hu[eé]spedes?)",
        text,
    )
    if match:
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return 2
    return 2


def _previous_tool_arguments(messages: list[dict]) -> dict:
    if len(messages) < 2:
        return {}
    calls = messages[-2].get("tool_calls") or []
    if not calls:
        return {}
    try:
        return json.loads(calls[0]["function"].get("arguments") or "{}")
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _grounded_policy_reply(result: str, spanish: bool, query: str) -> str:
    topic = query.lower()
    if "cancel" in topic:
        if spanish:
            return "Puede cancelar sin cargo hasta las 6:00 PM, hora local del hotel, dos días antes de la llegada. Las tarifas promocionales prepagadas no son reembolsables."
        return "You may cancel without charge until 6:00 PM local hotel time two days before arrival. Prepaid promotional rates are non-refundable."
    if "parking" in topic or "estacionamiento" in topic:
        if spanish:
            return "El estacionamiento cuesta $28 por noche y el servicio de valet cuesta $42 por noche."
        return "Self-parking is $28 per night, and valet parking is $42 per night."
    if "pet" in topic or "dog" in topic or "mascota" in topic:
        if spanish:
            return "Se permiten hasta dos perros por habitación, con un límite de 50 libras por perro y una tarifa de limpieza de $75 por estancia."
        return "Up to two dogs are allowed per room, with a 50-pound limit per dog and a $75 cleaning fee per stay."
    if "breakfast" in topic or "desayuno" in topic:
        if spanish:
            return "El desayuno se sirve de 6:30 AM a 10:30 AM y solo está incluido cuando la tarifa lo indica."
        return "Breakfast is served from 6:30 AM to 10:30 AM and is included only when the selected rate says so."
    if spanish:
        return "Encontré la política de Vera Hotel y puedo ayudarle con los detalles de su reserva."
    return "I found the relevant Vera Hotel policy and can help apply it to your reservation."


class SpeechOverrideProvider:
    """Composes a base LLM provider (groq/openai) with a Deepgram speech backend.

    Deepgram has no LLM, so `chat()` always goes to the base provider; STT and/or
    TTS are routed to Deepgram per the STT_PROVIDER / TTS_PROVIDER flags. This is
    how a two-vendor stack (e.g. Groq LLM + Deepgram speech) stays behind the one
    provider interface the agent and servers already consume.
    """

    def __init__(self, base, speech, *, use_stt: bool, use_tts: bool):
        self._base = base
        self._speech = speech
        self._use_stt = use_stt
        self._use_tts = use_tts
        # Name reflects which stages Deepgram actually handles (e.g. "groq+dg-stt").
        stages = [label for label, on in (("dg-stt", use_stt), ("dg-tts", use_tts)) if on]
        self.name = "+".join([base.name, *stages])
        # Mirror the base LLM attributes that telemetry / UI read via getattr.
        self.llm_model = base.llm_model
        self.tts_backend = getattr(base, "tts_backend", "provider")
        # Stage-model labels reflect who actually handles each stage.
        self.stt_model = speech.stt_model if use_stt else getattr(base, "stt_model", "unknown")
        self.tts_model = speech.tts_model if use_tts else getattr(base, "tts_model", "unknown")
        self.tts_voice = speech.tts_voice if use_tts else getattr(base, "tts_voice", "unknown")

    def chat(self, messages, tools=None, tool_choice=None):
        return self._base.chat(messages, tools=tools, tool_choice=tool_choice)

    def transcribe(self, pcm_int16, sample_rate: int = 16000):
        target = self._speech if self._use_stt else self._base
        return target.transcribe(pcm_int16, sample_rate)

    def transcribe_media(self, audio, content_type: str = ""):
        target = self._speech if self._use_stt else self._base
        return target.transcribe_media(audio, content_type)

    def synthesize(self, text, language=None):
        # Deepgram TTS only when cloud TTS is active; TTS_BACKEND=system stays local.
        # `language` lets Deepgram pick the native EN/ES voice per turn.
        if self._use_tts and self.tts_backend == "provider":
            return self._speech.synthesize(text, language)
        return self._base.synthesize(text, language)


def make_provider(name: str | None = None):
    """Factory: MockProvider for PROVIDER=mock, else a live Provider  -  optionally
    wrapped so STT_PROVIDER/TTS_PROVIDER=deepgram route those stages to Deepgram."""
    name = (name or os.getenv("PROVIDER", "groq")).lower()
    if name == "mock":
        return MockProvider()  # overrides ignored: offline path stays key-free

    # Validate speech overrides BEFORE building anything (fail fast, no key needed).
    stt_override = os.getenv("STT_PROVIDER", "").strip().lower()
    tts_override = os.getenv("TTS_PROVIDER", "").strip().lower()
    for label, value in (("STT_PROVIDER", stt_override), ("TTS_PROVIDER", tts_override)):
        if value and value != "deepgram":
            raise ValueError(f"Unknown {label}={value!r}; only 'deepgram' is supported")
    use_stt = stt_override == "deepgram"
    use_tts = tts_override == "deepgram"

    base = Provider(name)
    if not (use_stt or use_tts):
        return base

    from deepgram_speech import DeepgramSpeech  # lazy: only when actually selected
    speech = DeepgramSpeech()
    return SpeechOverrideProvider(base, speech, use_stt=use_stt, use_tts=use_tts)
