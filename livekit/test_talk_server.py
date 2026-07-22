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


class StreamingVoiceAgentTests(unittest.TestCase):
    """The NDJSON streaming twin (_stream_voice_agent_reply) via the emit seam:
    no HTTP, no sockets -- events are captured in a list; a scripted fake agent
    drives on_delta/on_reset exactly as agent.respond would."""

    def _session(self, provider, deltas, *, resets_before=None, reply="", action=None):
        """Install a fake session whose respond() streams `deltas` (with an
        optional on_reset before delta index N) and returns (reply, action)."""
        import threading

        import talk_server

        class FakeSessionAgent:
            def __init__(self):
                self.provider = provider
                self.last_sources = []
                self.current_language = "en"
                self.current_locale = "en-US"

            def respond(self, text, trace=None, on_delta=None, on_reset=None):
                for i, delta in enumerate(deltas):
                    if resets_before and i in resets_before and on_reset:
                        on_reset()
                    if on_delta:
                        on_delta(delta)
                return reply, action

        session = f"test-stream-{id(provider)}"
        talk_server._agent_sessions[session] = FakeSessionAgent()
        talk_server._session_locks[session] = threading.Lock()
        return session

    def _run(self, session, emit=None):
        from talk_server import _reset_session, _stream_voice_agent_reply

        events = []
        collect = emit or (lambda event: (events.append(event), True)[1])
        try:
            final = _stream_voice_agent_reply(
                b"audio", "audio/webm", session, "turn-1", False, emit=collect,
            )
        finally:
            _reset_session(session)
        return events, final

    class _StreamProvider:
        name = "openai+dg-tts"
        llm_model = "llm"
        stt_model = "stt-model"
        tts_backend = "provider"
        tts_model = "aura-2-luna-en"
        tts_voice = "aura-2-luna-en"

        def __init__(self, fail_on=()):
            self.fail_on = set(fail_on)
            self.synth_calls = 0

        def transcribe_media(self, audio, content_type=""):
            return "I need a room for two"

        def synthesize(self, text, language=None):
            self.synth_calls += 1
            if self.synth_calls in self.fail_on:
                raise RuntimeError("tts vendor error")
            return b"RIFFwav" + str(self.synth_calls).encode()

        def voice_for(self, language=None):
            return "aura-2-luna-en"

    def test_happy_path_streams_sentences_then_final(self):
        import base64 as b64

        provider = self._StreamProvider()
        session = self._session(
            provider, ["Sure. ", "We have a king. "],
            reply="Sure. We have a king.",
        )
        events, final = self._run(session)

        kinds = [event["type"] for event in events]
        self.assertEqual(kinds, ["transcript", "sentence", "sentence", "final"])
        self.assertEqual(events[1]["text"], "Sure.")
        self.assertTrue(b64.b64decode(events[1]["audioBase64"]).startswith(b"RIFF"))
        self.assertEqual(final["pipeline"], "streaming")   # audio actually streamed
        self.assertEqual(final["reply"], "Sure. We have a king.")
        self.assertIn("trace", final)
        self.assertEqual(final["sentences"], 2)

    def test_single_sentence_tts_failure_does_not_degrade_turn(self):
        provider = self._StreamProvider(fail_on={2})
        session = self._session(
            provider, ["One. ", "Two. "], reply="One. Two.",
        )
        events, final = self._run(session)
        sentences = [event for event in events if event["type"] == "sentence"]
        self.assertTrue(sentences[0]["audioBase64"])       # first synthesized
        self.assertEqual(sentences[1]["audioBase64"], "")  # failed -> empty, no raise
        self.assertEqual(final["pipeline"], "streaming")   # one chunk still streamed
        self.assertNotEqual(final.get("action"), "transfer")  # NOT degraded

    def test_all_tts_failures_fall_back_to_browser_voice(self):
        provider = self._StreamProvider(fail_on={1, 2})
        session = self._session(provider, ["One. ", "Two. "], reply="One. Two.")
        events, final = self._run(session)
        self.assertTrue(all(
            event["audioBase64"] == "" for event in events if event["type"] == "sentence"
        ))
        self.assertTrue(final["ttsFallback"])
        self.assertEqual(final["ttsBackend"], "browser")
        self.assertEqual(final["pipeline"], "batch")       # honest badge: nothing streamed

    def test_client_gone_skips_tts_but_completes_turn(self):
        provider = self._StreamProvider()
        session = self._session(
            provider, ["One. ", "Two. ", "Three. "], reply="One. Two. Three.",
        )
        emitted = []

        def dying_emit(event):
            emitted.append(event)
            return len(emitted) < 3  # client disappears after the 2nd event

        events, final = self._run(session, emit=dying_emit)
        self.assertEqual(provider.synth_calls, 2)  # TTS stopped after client left
        self.assertEqual(final["reply"], "One. Two. Three.")  # turn still completed

    def test_reset_drops_preamble_segment_and_degrade_reply_is_sent(self):
        provider = self._StreamProvider()
        session = self._session(
            provider, ["One moment. ", "Booked! "],
            resets_before={1},                      # tool boundary after the preamble
            reply="Booked!",
        )
        events, _final = self._run(session)
        kinds = [event["type"] for event in events]
        self.assertIn("reset", kinds)
        reset_event = next(event for event in events if event["type"] == "reset")
        self.assertEqual(reset_event["segment"], 1)
        post_reset = [event for event in events
                      if event["type"] == "sentence" and event["segment"] == 1]
        self.assertEqual(post_reset[0]["text"], "Booked!")

        # Degrade guarantee: a respond() that streams NOTHING still sends the reply.
        quiet = self._StreamProvider()
        session2 = self._session(quiet, [], reply="Let me transfer you.", action="transfer")
        events2, final2 = self._run(session2)
        sentences2 = [event for event in events2 if event["type"] == "sentence"]
        self.assertEqual(sentences2[0]["text"], "Let me transfer you.")
        self.assertEqual(final2["action"], "transfer")

    def test_ignored_turn_emits_single_final(self):
        provider = self._StreamProvider()
        provider.transcribe_media = lambda audio, content_type="": "..."  # noise
        session = self._session(provider, ["never"], reply="never")
        events, final = self._run(session)
        self.assertEqual([event["type"] for event in events], ["final"])
        self.assertTrue(final["ignored"])
        self.assertEqual(final["ignoreReason"], "noise_or_hallucination")

    def test_first_audio_event_fires_once_on_first_audible_sentence(self):
        # First sentence's TTS fails (no audio), so tts.first_audio must fire on the
        # SECOND (first audible) sentence, exactly once -- locks the
        # `audio_b64 and not state["first_audio"]` guard against a no-audio first sentence.
        provider = self._StreamProvider(fail_on={1})
        session = self._session(
            provider, ["Hello. ", "We have a king. "],
            reply="Hello. We have a king.",
        )
        events, final = self._run(session)

        first_audio = [e for e in final["trace"]["events"] if e["name"] == "tts.first_audio"]
        self.assertEqual(len(first_audio), 1)
        sentences = [e for e in events if e["type"] == "sentence"]
        self.assertEqual(sentences[0]["audioBase64"], "")   # first sentence's TTS failed
        self.assertTrue(sentences[1]["audioBase64"])         # second carried the first audio

    def test_streaming_events_carry_progressive_timings(self):
        # transcript/sentence events include the server's RUNNING timings snapshot so
        # the client fills metric tiles during the stream (not only at `final`):
        # stt is present from the transcript event on, and tts accumulates per sentence.
        provider = self._StreamProvider()
        session = self._session(
            provider, ["Sure. ", "We have a king. "],
            reply="Sure. We have a king.",
        )
        events, final = self._run(session)

        transcript = events[0]
        self.assertIn("stt", transcript["timings"])
        sentences = [e for e in events if e["type"] == "sentence"]
        self.assertIn("tts", sentences[0]["timings"])
        self.assertGreaterEqual(
            sentences[1]["timings"]["tts"], sentences[0]["timings"]["tts"],
        )
        # `total` never appears mid-stream -- the Server tile is final-only.
        self.assertNotIn("total", sentences[0]["timings"])

    def test_gate_stays_batch_when_streaming_off_or_incapable(self):
        import os
        from unittest.mock import patch

        from talk_server import _browser_streaming_enabled

        class CapableAgent:
            class provider:  # noqa: N801 - attribute stand-in
                tts_backend = "provider"

                @staticmethod
                def chat_stream(*args, **kwargs):
                    return None

        with patch.dict(os.environ, {"STREAMING": "off"}):
            self.assertFalse(_browser_streaming_enabled(CapableAgent()))
        with patch.dict(os.environ, {"STREAMING": "on"}):
            self.assertTrue(_browser_streaming_enabled(CapableAgent()))

            class MockLikeAgent:
                class provider:  # noqa: N801
                    tts_backend = "print"
            self.assertFalse(_browser_streaming_enabled(MockLikeAgent()))


class BargeInEchoTests(unittest.TestCase):
    """Playback-echo suppression on barge-in: normalization must catch smart_format
    punctuation, and a courtesy phrase is dropped ONLY when the agent actually said it."""

    def test_normalize_strips_all_punctuation_including_smart_format(self):
        from talk_server import _normalize_echo
        self.assertEqual(_normalize_echo("Thanks!"), "thanks")
        self.assertEqual(_normalize_echo("Thank you."), "thank you")
        self.assertEqual(_normalize_echo("Alright?"), "alright")
        self.assertEqual(_normalize_echo("You're welcome!"), "youre welcome")

    def test_probable_echo_catches_exclamation_and_question(self):
        from talk_server import _is_probable_playback_echo
        self.assertTrue(_is_probable_playback_echo("Thanks!"))       # smart_format restores !
        self.assertTrue(_is_probable_playback_echo("Thank you."))
        self.assertFalse(_is_probable_playback_echo("Do you have a room?"))

    def test_last_agent_reply_returns_latest_assistant_content(self):
        from talk_server import _last_agent_reply
        agent = type("A", (), {"messages": [
            {"role": "assistant", "content": "First reply."},
            {"role": "user", "content": "book"},
            {"role": "assistant", "content": "Sure, thank you!"},
        ]})()
        self.assertEqual(_last_agent_reply(agent), "Sure, thank you!")

    def test_barge_in_echo_matching_reply_is_suppressed(self):
        import threading

        import talk_server

        class EchoProvider:
            name = "openai"; llm_model = "x"; stt_model = "s"
            tts_backend = "provider"; tts_model = "t"; tts_voice = "v"

            def transcribe_media(self, audio, content_type=""):
                return "Thank you!"  # smart_format '!' echo of what the agent just said

        class FakeSessionAgent:
            def __init__(self):
                self.provider = EchoProvider()
                self.last_sources = []
                self.current_language = "en"
                self.current_locale = "en-US"
                self.messages = [{"role": "assistant", "content": "Booked. Thank you!"}]

        session = "test-echo-suppress"
        talk_server._agent_sessions[session] = FakeSessionAgent()
        talk_server._session_locks[session] = threading.Lock()
        try:
            payload = _voice_agent_reply(b"audio", "audio/webm", session, "turn-1", True)
        finally:
            _reset_session(session)
        self.assertTrue(payload.get("ignored"))
        self.assertEqual(payload.get("ignoreReason"), "probable_playback_echo")

    def test_barge_in_genuine_courtesy_not_in_reply_passes_through(self):
        from talk_server import _transcribe_turn

        class CourtesyProvider:
            name = "openai"; llm_model = "x"; stt_model = "s"
            tts_backend = "provider"; tts_model = "t"; tts_voice = "v"

            def transcribe_media(self, audio, content_type=""):
                return "Thanks"

        class FakeSessionAgent:
            provider = CourtesyProvider()
            last_sources = []
            current_language = "en"
            current_locale = "en-US"
            # The agent never said "thanks" -> a genuine "Thanks" interruption must pass.
            messages = [{"role": "assistant", "content": "We have a Standard Queen for $189."}]

        transcript, ignored = _transcribe_turn(
            FakeSessionAgent(), FakeTrace(), b"audio", "audio/webm", True,
        )
        self.assertIsNone(ignored)
        self.assertEqual(transcript, "Thanks")


class SessionRegistryTests(unittest.TestCase):
    """LRU eviction bounds the per-X-Session-ID Agent registry; an in-flight
    session (lock held) is never evicted."""

    def setUp(self):
        import talk_server
        self.ts = talk_server
        self._saved = (dict(talk_server._agent_sessions),
                       dict(talk_server._session_locks), talk_server.MAX_SESSIONS)
        talk_server._agent_sessions.clear()
        talk_server._session_locks.clear()

    def tearDown(self):
        self.ts._agent_sessions.clear()
        self.ts._agent_sessions.update(self._saved[0])
        self.ts._session_locks.clear()
        self.ts._session_locks.update(self._saved[1])
        self.ts.MAX_SESSIONS = self._saved[2]

    def test_lru_eviction_bounds_the_registry(self):
        from unittest.mock import patch
        with patch.object(self.ts, "_new_agent", lambda: object()), \
                patch.object(self.ts, "MAX_SESSIONS", 3):
            for i in range(6):
                self.ts._get_session(f"s{i}")
            self.assertLessEqual(len(self.ts._agent_sessions), 3)
            self.assertNotIn("s0", self.ts._agent_sessions)  # oldest evicted
            self.assertIn("s5", self.ts._agent_sessions)     # newest kept

    def test_in_flight_session_is_never_evicted(self):
        from unittest.mock import patch
        with patch.object(self.ts, "_new_agent", lambda: object()), \
                patch.object(self.ts, "MAX_SESSIONS", 2):
            _agent, lock = self.ts._get_session("busy")
            lock.acquire()  # a turn is in flight
            try:
                for i in range(4):
                    self.ts._get_session(f"s{i}")
                self.assertIn("busy", self.ts._agent_sessions)
            finally:
                lock.release()


class RequestGuardTests(unittest.TestCase):
    """_read_length bounds/validates Content-Length, and the static handler
    resolves traversal attempts to a non-source path (source-disclosure fix)."""

    def _handler(self):
        import http.client

        from talk_server import Handler
        h = Handler.__new__(Handler)
        h.headers = http.client.HTTPMessage()
        h.sent = []
        h._send_json = lambda payload, status=200: h.sent.append((status, payload))
        return h

    def test_valid_content_length_accepted(self):
        h = self._handler()
        h.headers["Content-Length"] = "2048"
        self.assertEqual(h._read_length(), 2048)
        self.assertEqual(h.sent, [])

    def test_oversized_content_length_rejected_413(self):
        h = self._handler()
        h.headers["Content-Length"] = str(999_000_000)
        self.assertIsNone(h._read_length())
        self.assertEqual(h.sent[0][0], 413)

    def test_nonnumeric_content_length_rejected_400(self):
        h = self._handler()
        h.headers["Content-Length"] = "not-a-number"
        self.assertIsNone(h._read_length())
        self.assertEqual(h.sent[0][0], 400)

    def test_traversal_cannot_escape_the_web_root(self):
        import os

        from talk_server import ROOT, Handler
        h = Handler.__new__(Handler)
        h.directory = str(ROOT)
        real_source = os.path.realpath(os.path.join(str(ROOT), "talk_server.py"))
        # "/web/../talk_server.py" -- normpath collapses it out of /web/ -- must NOT
        # resolve to the real source file.
        escaped = os.path.realpath(h.translate_path("/web/../talk_server.py"))
        self.assertNotEqual(escaped, real_source)
        # a legit asset still resolves under web/
        web_root = os.path.realpath(os.path.join(str(ROOT), "web"))
        legit = os.path.realpath(h.translate_path("/web/index.html"))
        self.assertTrue(legit.startswith(web_root + os.sep))


if __name__ == "__main__":
    unittest.main()
