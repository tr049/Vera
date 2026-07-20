"""Offline tests for browser TTS payload selection + the voice-agent STT seam."""

from __future__ import annotations

import base64
import os
import unittest
from contextlib import contextmanager

os.environ["PROVIDER"] = "mock"  # keep the voice-agent test offline

from talk_server import _browser_tts_payload, _reset_session, _voice_agent_reply


class FakeTrace:
    def __init__(self):
        self.events = []

    @contextmanager
    def span(self, name, **attributes):
        yield

    def event(self, name, **attributes):
        self.events.append((name, attributes))


class FakeProvider:
    name = "openai"
    tts_model = "tts-test"
    tts_voice = "voice-test"

    def __init__(self, backend="provider", error=None):
        self.tts_backend = backend
        self.error = error
        self.calls = []

    def synthesize(self, text, language=None):
        self.calls.append(text)
        if self.error:
            raise self.error
        return b"RIFFtest-wave"


class FakeAgent:
    def __init__(self, provider):
        self.provider = provider


class BrowserTtsPayloadTests(unittest.TestCase):
    def test_provider_backend_returns_audio(self):
        provider = FakeProvider()
        payload = _browser_tts_payload(FakeAgent(provider), FakeTrace(), "Hello")

        self.assertEqual(payload["ttsBackend"], "provider")
        self.assertEqual(base64.b64decode(payload["audioBase64"]), b"RIFFtest-wave")
        self.assertEqual(payload["ttsVoice"], "voice-test")
        self.assertEqual(provider.calls, ["Hello"])

    def test_system_backend_selects_browser_voice_without_provider_call(self):
        provider = FakeProvider(backend="system")
        payload = _browser_tts_payload(FakeAgent(provider), FakeTrace(), "Hello")

        self.assertEqual(payload, {"ttsBackend": "browser"})
        self.assertEqual(provider.calls, [])

    def test_provider_failure_falls_back_without_exposing_error(self):
        provider = FakeProvider(error=RuntimeError("secret provider response"))
        trace = FakeTrace()
        payload = _browser_tts_payload(FakeAgent(provider), trace, "Hello")

        self.assertEqual(payload, {"ttsBackend": "browser", "ttsFallback": True})
        self.assertEqual(trace.events[0][0], "tts.fallback")
        self.assertNotIn("secret provider response", str(payload))

    def test_voice_for_labels_per_turn_voice_without_collapsing_model(self):
        # Deepgram TTS: the label must show the ACTUAL per-turn voice (celeste for a
        # Spanish turn) while ttsModel stays the configured model -- never collapse
        # model onto the voice (that mislabelled the STT-only two-vendor stack).
        class VoiceForProvider(FakeProvider):
            tts_model = "aura-2-luna-en"
            tts_voice = "aura-2-luna-en"

            def voice_for(self, language=None):
                return {"es": "aura-2-celeste-es"}.get(language, "aura-2-luna-en")

        agent = FakeAgent(VoiceForProvider())
        agent.current_language = "es"
        payload = _browser_tts_payload(agent, FakeTrace(), "Hola")

        self.assertEqual(payload["ttsVoice"], "aura-2-celeste-es")  # what the caller heard
        self.assertEqual(payload["ttsModel"], "aura-2-luna-en")     # real model, not the voice

    def test_voice_for_stt_only_stack_keeps_base_model_and_voice(self):
        # STT-only Deepgram: TTS runs on the base provider, so voice_for returns the
        # base voice and ttsModel stays the base model -- neither is mislabelled.
        class SttOnlyProvider(FakeProvider):
            tts_model = "gpt-4o-mini-tts"
            tts_voice = "alloy"

            def voice_for(self, language=None):
                return "alloy"  # SpeechOverrideProvider returns the base voice here

        payload = _browser_tts_payload(FakeAgent(SttOnlyProvider()), FakeTrace(), "Hi")

        self.assertEqual(payload["ttsModel"], "gpt-4o-mini-tts")  # not 'alloy'
        self.assertEqual(payload["ttsVoice"], "alloy")


class VoiceAgentReplyTests(unittest.TestCase):
    """Covers the refactored STT seam: _voice_agent_reply must transcribe via the
    provider's transcribe_media() (not an OpenAI-SDK reach-in) for any backend."""

    def test_mock_browser_turn_transcribes_and_replies(self):
        session = "test-voice-seam"
        _reset_session(session)
        try:
            payload = _voice_agent_reply(
                b"pretend-webm-audio", "audio/webm", session, "turn-1", False,
            )
        finally:
            _reset_session(session)
        # Transcript comes from the mock's scripted STT via transcribe_media,
        # proving the uniform seam works with no OpenAI-specific bypass.
        self.assertIn("room", payload["transcript"].lower())
        self.assertTrue(payload["reply"])
        self.assertEqual(payload["ttsBackend"], "browser")  # mock -> browser TTS

    def test_stt_error_degrades_instead_of_500(self):
        import threading

        import talk_server

        class RaisingProvider:
            name = "openai"
            llm_model = "gpt-4.1-mini"
            stt_model = "deepgram-nova-3"
            tts_backend = "provider"
            tts_model = "aura-2"
            tts_voice = "aura-2"

            def transcribe_media(self, audio, content_type=""):
                raise RuntimeError("401 Unauthorized: bad DEEPGRAM_API_KEY")

        class FakeSessionAgent:
            def __init__(self):
                self.provider = RaisingProvider()
                self.last_sources = []
                self.current_language = "en"
                self.current_locale = "en-US"

        session = "test-stt-error"
        talk_server._agent_sessions[session] = FakeSessionAgent()
        talk_server._session_locks[session] = threading.Lock()
        try:
            # Must NOT raise (would 500 + leak str(exc)); degrades gracefully instead.
            payload = _voice_agent_reply(b"audio", "audio/webm", session, "turn-1", False)
        finally:
            _reset_session(session)
        self.assertTrue(payload.get("ignored"))
        self.assertEqual(payload.get("ignoreReason"), "stt_error")
        self.assertNotIn("Unauthorized", str(payload))  # raw error not leaked

    def test_noise_transcript_is_suppressed(self):
        import threading

        import talk_server

        class NoiseProvider:
            name = "openai"; llm_model = "x"; stt_model = "s"
            tts_backend = "provider"; tts_model = "t"; tts_voice = "v"

            def transcribe_media(self, audio, content_type=""):
                return "..."  # punctuation-only -> is_noise_transcript True

        class FakeSessionAgent:
            def __init__(self):
                self.provider = NoiseProvider()
                self.last_sources = []
                self.current_language = "en"
                self.current_locale = "en-US"

        session = "test-noise-suppress"
        talk_server._agent_sessions[session] = FakeSessionAgent()
        talk_server._session_locks[session] = threading.Lock()
        try:
            payload = _voice_agent_reply(b"audio", "audio/webm", session, "turn-1", False)
        finally:
            _reset_session(session)
        self.assertTrue(payload.get("ignored"))
        self.assertEqual(payload.get("ignoreReason"), "noise_or_hallucination")


if __name__ == "__main__":
    unittest.main()
