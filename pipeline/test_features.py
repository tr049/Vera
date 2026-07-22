"""Focused offline tests for routing, grounding, telemetry, and capacity."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ["PROVIDER"] = "mock"
os.environ.setdefault("TTS_BACKEND", "print")

from agent import Agent, explicit_language_request, required_tool_for, run_tool
from deepgram_speech import DeepgramSpeech, _first_transcript
from inventory import active_bookings, reset_inventory
from knowledge import search_hotel_knowledge
from providers import (
    MockProvider,
    Provider,
    SpeechOverrideProvider,
    _env_or_default,
    _mk_tool,
    is_noise_transcript,
    is_stt_hallucination,
    make_provider,
)
from router import AgentRouter
from scale_check import estimate_capacity
from streaming import SentenceBuffer, streaming_enabled
from telemetry import TurnTrace


class RouterTests(unittest.TestCase):
    def test_tool_selected_language_persists(self):
        router = AgentRouter()
        self.assertEqual(router.set_language("es").language, "es")
        self.assertEqual(router.route().language, "es")
        self.assertEqual(router.set_language("en").language, "en")

    def test_language_switch_intent_uses_control_tool(self):
        agent = Agent(make_provider("mock"))
        spanish_trace = TurnTrace(session_id="test", turn_id="spanish")
        agent.respond("Can you please speak in Spanish?", trace=spanish_trace)
        english_trace = TurnTrace(session_id="test", turn_id="english")
        reply, _ = agent.respond("Let me switch back to English.", trace=english_trace)

        self.assertEqual(agent.current_language, "en")
        self.assertIn("continue in English", reply)
        requested = [
            event["attributes"].get("tool")
            for event in english_trace.events
            if event["name"] == "tool.requested"
        ]
        self.assertEqual(requested, ["set_language"])
        self.assertIn(
            "router.language_changed",
            [event["name"] for event in english_trace.events],
        )

    def test_polite_phrasing_forces_deterministic_language_switch(self):
        # "Could you please tell me in english" must flip the router even when the
        # LLM would not have called set_language on its own -- the switch is FORCED
        # as the first tool call (live models sometimes just answer in English).
        agent = Agent(make_provider("mock"))
        agent.respond("Can you please speak in Spanish?",
                      trace=TurnTrace(session_id="test", turn_id="to-es"))
        self.assertEqual(agent.current_language, "es")
        trace = TurnTrace(session_id="test", turn_id="back-to-en")
        agent.respond("Could you please tell me in english", trace=trace)
        self.assertEqual(agent.current_language, "en")
        routed = [event["attributes"] for event in trace.events
                  if event["name"] == "tool.route_selected"]
        self.assertEqual(routed[0]["tool"], "set_language")
        self.assertEqual(routed[0]["reason"], "explicit_language_request")

    def test_language_switch_requires_a_switch_cue_not_a_bare_mention(self):
        # requested_language_switch must fire only on directional intent, never on a
        # bare language name -- otherwise an in-scope question or a negation gets
        # hijacked into a wrong-way switch (regression the pre-ship audit caught).
        from agent import requested_language_switch
        # positive: real switch cues carry their own target
        self.assertEqual(requested_language_switch("please tell me in Spanish", "en"), "es")
        self.assertEqual(requested_language_switch("could you speak English again", "es"), "en")
        self.assertEqual(requested_language_switch("háblame en español", "en"), "es")
        # negative: bare mentions / questions must NOT switch
        self.assertIsNone(requested_language_switch(
            "Do you have Spanish-speaking staff at the front desk?", "en"))
        self.assertIsNone(requested_language_switch("I do not want Spanish", "en"))
        self.assertIsNone(requested_language_switch("Is English breakfast included?", "es"))
        # already in the requested language -> no-op
        self.assertIsNone(requested_language_switch("please continue in English", "en"))

    def test_bare_language_mention_does_not_hijack_an_in_scope_question(self):
        # End-to-end: an English question that merely contains "Spanish" stays in
        # English and is answered, not force-switched to Spanish.
        agent = Agent(make_provider("mock"))
        trace = TurnTrace(session_id="test", turn_id="staff-q")
        agent.respond("Do you have Spanish-speaking staff at the front desk?", trace=trace)
        self.assertEqual(agent.current_language, "en")
        routed = [event["attributes"].get("tool") for event in trace.events
                  if event["name"] == "tool.route_selected"]
        self.assertNotIn("set_language", routed)

    def test_language_change_requires_explicit_target_name(self):
        self.assertTrue(explicit_language_request("Switch back to English", "en"))
        self.assertTrue(explicit_language_request("Por favor, habla español", "es"))
        self.assertFalse(explicit_language_request("¡Gracias!", "es"))

    def test_overeager_language_tool_cannot_change_state(self):
        class OvereagerProvider(MockProvider):
            def chat(self, messages, tools=None, tool_choice=None):
                if messages[-1].get("role") == "user":
                    return _mk_tool("set_language", {"language": "es"})
                return super().chat(messages, tools=tools, tool_choice=tool_choice)

        agent = Agent(OvereagerProvider())
        trace = TurnTrace(session_id="test", turn_id="courtesy")
        agent.respond("¡Gracias!", trace=trace)

        self.assertEqual(agent.current_language, "en")
        self.assertIn(
            "router.language_change_rejected",
            [event["name"] for event in trace.events],
        )


class ProviderConfigurationTests(unittest.TestCase):
    def test_blank_model_override_uses_provider_default(self):
        with patch.dict(os.environ, {"LLM_MODEL": ""}):
            self.assertEqual(_env_or_default("LLM_MODEL", "gpt-4o-mini"), "gpt-4o-mini")

    def test_comment_only_model_override_uses_provider_default(self):
        with patch.dict(os.environ, {"LLM_MODEL": "# example model"}):
            self.assertEqual(_env_or_default("LLM_MODEL", "gpt-4o-mini"), "gpt-4o-mini")

    def test_explicit_model_override_is_preserved(self):
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-4.1-mini"}):
            self.assertEqual(_env_or_default("LLM_MODEL", "gpt-4o-mini"), "gpt-4.1-mini")


class RetrievalTests(unittest.TestCase):
    def test_english_policy_returns_precise_source(self):
        result = search_hotel_knowledge("What is the cancellation policy?")
        self.assertEqual(result["sources"], ["hotel_policies.md#Cancellation"])

    def test_spanish_query_expands_to_english_knowledge(self):
        result = search_hotel_knowledge("¿Cuál es la política de mascotas?")
        self.assertEqual(result["sources"], ["hotel_policies.md#Pets"])

    def test_policy_intent_requires_grounding_tool(self):
        self.assertEqual(
            required_tool_for("What does the cancellation policy look like?"),
            "search_hotel_knowledge",
        )

    def test_cancellation_action_is_not_misrouted_to_rag(self):
        self.assertIsNone(required_tool_for("Please cancel my reservation"))

    def test_noisy_spanish_pet_policy_transcript_routes_to_rag(self):
        self.assertEqual(
            required_tool_for("Fiol es la politista di maskotas."),
            "search_hotel_knowledge",
        )

    def test_check_in_as_booking_verb_is_not_misrouted_to_rag(self):
        # "check in <date>" is a booking, not a policy question; it must reach the
        # LLM (and check_availability), not be force-routed to search_hotel_knowledge.
        self.assertIsNone(required_tool_for("I want to check in this weekend for two guests"))
        self.assertIsNone(required_tool_for("Can I check in on Friday?"))

    def test_check_in_time_question_still_routes_to_rag(self):
        for question in ("What time is check-in?", "What's the check-out time?",
                         "When is check-in?"):
            self.assertEqual(
                required_tool_for(question), "search_hotel_knowledge", question,
            )

    def test_forced_tool_choice_is_sent_on_first_model_call(self):
        class RecordingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.tool_choices = []

            def chat(self, messages, tools=None, tool_choice=None):
                self.tool_choices.append(tool_choice)
                return super().chat(messages, tools=tools, tool_choice=tool_choice)

        provider = RecordingProvider()
        agent = Agent(provider)
        trace = TurnTrace(session_id="test", turn_id="forced-rag")
        reply, _ = agent.respond("What is the cancellation policy?", trace=trace)

        self.assertIn("6:00 PM", reply)
        self.assertEqual(
            provider.tool_choices[0],
            {"type": "function", "function": {"name": "search_hotel_knowledge"}},
        )
        self.assertIsNone(provider.tool_choices[1])
        self.assertIn("tool.route_selected", [event["name"] for event in trace.events])


class TelemetryTests(unittest.TestCase):
    def test_tool_and_language_events_are_visible(self):
        agent = Agent(make_provider("mock"))
        trace = TurnTrace(session_id="test", turn_id="policy")
        reply, action = agent.respond("What is the pet policy?", trace=trace)
        payload = trace.finish(action=action, sources=agent.last_sources)
        event_names = [event["name"] for event in payload["events"]]
        requested_tools = [
            event["attributes"].get("tool")
            for event in payload["events"]
            if event["name"] == "tool.requested"
        ]
        self.assertIn("two dogs", reply)
        self.assertIn("retrieval.completed", event_names)
        self.assertEqual(requested_tools, ["search_hotel_knowledge"])
        self.assertEqual(payload["attributes"]["language"], "en")

    def test_sensitive_tool_arguments_are_redacted(self):
        trace = TurnTrace(session_id="test", turn_id="redaction")
        trace.event("tool.requested", arguments={
            "guest_name": "Priya Shah",
            "contact": "priya@example.com",
            "check_in": "August 12",
        })
        attributes = trace.events[0]["attributes"]["arguments"]
        self.assertEqual(attributes["guest_name"], "[REDACTED]")
        self.assertEqual(attributes["contact"], "[REDACTED]")
        self.assertEqual(attributes["check_in"], "August 12")


class ScaleTests(unittest.TestCase):
    def test_one_million_dau_example(self):
        result = estimate_capacity(
            dau=1_000_000,
            calls_per_dau=0.25,
            duration_minutes=4,
            turns_per_minute=3,
            peak_factor=8,
            sessions_per_worker=40,
            headroom=0.30,
            cost_per_minute=0,
        )
        self.assertAlmostEqual(result["peakConcurrency"], 5555.6)
        self.assertEqual(result["workers"], 181)


class BookingToolTests(unittest.TestCase):
    def setUp(self):
        reset_inventory()

    def tearDown(self):
        reset_inventory()

    def test_available_rooms_are_listed_for_a_normal_party(self):
        result = run_tool("check_availability", {
            "guests": 2, "room_type": "standard",
            "check_in": "August 12", "check_out": "August 14",
        })
        self.assertIn("Standard Queen", result["result"])
        self.assertNotIn("action", result)

    def test_oversized_party_returns_transfer_fallback(self):
        result = run_tool("check_availability", {
            "guests": 9,
            "check_in": "August 12", "check_out": "August 14",
        })
        self.assertIn("No matching rooms", result["result"])
        self.assertIn("front desk", result["result"].lower())

    def test_create_booking_returns_confirmation_code(self):
        result = run_tool("create_booking", {
            "guest_name": "Priya Shah", "contact": "priya@example.com",
            "check_in": "August 12", "check_out": "August 14",
            "guests": 2, "room_type": "standard",
        })
        self.assertIn("VH-4827", result["result"])


class InventoryTests(unittest.TestCase):
    def setUp(self):
        reset_inventory()

    def tearDown(self):
        reset_inventory()

    def _book(self, room_type="standard", guest="Ada", contact="ada@example.com",
              check_in="August 12", check_out="August 14"):
        return run_tool("create_booking", {
            "room_type": room_type, "guest_name": guest, "contact": contact,
            "guests": 2, "check_in": check_in, "check_out": check_out,
        })

    def _availability(self, room_type="standard", check_in="August 12",
                      check_out="August 14", guests=2):
        return run_tool("check_availability", {
            "guests": guests, "room_type": room_type,
            "check_in": check_in, "check_out": check_out,
        })

    def test_booked_room_disappears_from_availability(self):
        self.assertIn("Standard Queen", self._availability()["result"])
        self._book()
        self.assertIn("No matching rooms", self._availability()["result"])

    def test_double_booking_same_room_and_dates_is_refused(self):
        first = self._book()
        self.assertIn("VH-4827", first["result"])
        second = self._book(guest="Bo", contact="bo@example.com")
        self.assertNotIn("Booking confirmed", second["result"])
        self.assertEqual(len(active_bookings()), 1)  # the refused one was not stored

    def test_confirmation_codes_increment_across_bookings(self):
        self.assertIn("VH-4827", self._book(room_type="standard")["result"])
        self.assertIn("VH-4828", self._book(room_type="king")["result"])

    def test_non_overlapping_dates_remain_available(self):
        self._book(check_in="August 12", check_out="August 14")
        later = self._availability(check_in="August 20", check_out="August 22")
        self.assertIn("Standard Queen", later["result"])

    def test_checkout_before_checkin_is_rejected(self):
        result = self._availability(check_in="August 14", check_out="August 12")
        self.assertNotIn("Available rooms", result["result"])
        self.assertIn("check-out", result["result"].lower())

    def test_negative_guest_count_is_rejected(self):
        result = run_tool("check_availability", {
            "guests": -1, "check_in": "August 12", "check_out": "August 14",
        })
        self.assertNotIn("Available rooms", result["result"])

    def test_create_booking_rejects_over_capacity(self):
        result = run_tool("create_booking", {
            "room_type": "standard", "guest_name": "A", "contact": "a@x.com",
            "guests": 5, "check_in": "August 12", "check_out": "August 14",
        })
        self.assertNotIn("Booking confirmed", result["result"])  # cap-2 room, 5 guests
        self.assertIn("holds up to", result["result"])
        self.assertEqual(len(active_bookings()), 0)  # nothing stored


class TranscriptFilterTests(unittest.TestCase):
    def test_hallucinated_greetings_and_echoes_are_flagged(self):
        for garbage in [
            "ChatGPT, ¿Cómo puedo ayudarte hoy?",   # the exact observed echo
            "How can I help you today?",
            "How can I assist you?",
            "¿Cómo puedo ayudarte?",
            "Thanks for watching!",
            "Please subscribe.",
        ]:
            self.assertTrue(is_stt_hallucination(garbage), garbage)

    def test_real_caller_turns_are_not_flagged(self):
        for real in [
            "Hello, I would like to make a reservation at Vera Hotel.",
            "I need a room for two guests.",
            "Yes, book it for Priya Shah.",
            "What is the cancellation policy?",
            "Goodbye",
            "Gracias, ¿pueden ayudarme con una reserva?",
        ]:
            self.assertFalse(is_stt_hallucination(real), real)

    def test_noise_and_hallucination_filters_are_independent(self):
        # empty/punctuation is noise, not a hallucination phrase
        self.assertTrue(is_noise_transcript("..."))
        self.assertFalse(is_stt_hallucination("..."))


class ResilienceTests(unittest.TestCase):
    def test_provider_error_degrades_to_transfer(self):
        class ExplodingProvider(MockProvider):
            def chat(self, messages, tools=None, tool_choice=None):
                raise RuntimeError("simulated provider outage")

        agent = Agent(ExplodingProvider())
        trace = TurnTrace(session_id="test", turn_id="provider-error")
        reply, action = agent.respond("I need a room for two guests.", trace=trace)

        self.assertEqual(action, "transfer")
        self.assertIn("front desk", reply.lower())
        event_names = [event["name"] for event in trace.events]
        self.assertIn("provider.error", event_names)
        self.assertIn("agent.degraded", event_names)

    def test_runaway_tool_loop_is_capped(self):
        class LoopingProvider(MockProvider):
            def chat(self, messages, tools=None, tool_choice=None):
                return _mk_tool("check_availability", {
                    "guests": 2, "room_type": "standard",
                    "check_in": "August 12", "check_out": "August 14",
                })

        agent = Agent(LoopingProvider())
        trace = TurnTrace(session_id="test", turn_id="runaway")
        reply, action = agent.respond("hello", trace=trace)

        self.assertEqual(action, "transfer")
        self.assertIn(
            "tool_loop.exhausted",
            [event["name"] for event in trace.events],
        )


class _StubBase:
    name = "groq"
    llm_model = "llm"
    tts_backend = "provider"
    stt_model = "base-stt"
    tts_model = "base-tts"
    tts_voice = "base-voice"

    def chat(self, messages, tools=None, tool_choice=None):
        return "chat"

    def chat_stream(self, messages, tools=None, tool_choice=None, on_delta=None):
        from types import SimpleNamespace as NS
        if on_delta:
            on_delta("chat")
        return NS(choices=[NS(message=NS(content="chat", tool_calls=None))])

    def transcribe(self, pcm, sample_rate=16000):
        return "base-transcribe"

    def transcribe_media(self, audio, content_type=""):
        return "base-media"

    def synthesize(self, text, language=None):
        return b"base-audio"


class _StubSpeech:
    name = "deepgram"
    stt_model = "nova-3"
    tts_model = "aura-2"
    tts_voice = "aura-2"

    def transcribe(self, pcm, sample_rate=16000):
        return "dg-transcribe"

    def transcribe_media(self, audio, content_type=""):
        return "dg-media"

    def synthesize(self, text, language=None):
        return b"dg-audio"

    def voice_for(self, language=None):
        return {"es": "aura-2-celeste-es"}.get(language, "aura-2-luna-en")


class DeepgramOverrideTests(unittest.TestCase):
    def test_mock_ignores_speech_overrides_and_needs_no_key(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "deepgram", "TTS_PROVIDER": "deepgram"}):
            os.environ.pop("DEEPGRAM_API_KEY", None)
            provider = make_provider("mock")
        self.assertIsInstance(provider, MockProvider)

    def test_unknown_stt_provider_raises_before_needing_a_key(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "whisper-cloud"}):
            with self.assertRaises(ValueError):
                make_provider("openai")

    def test_deepgram_speech_requires_api_key(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("DEEPGRAM_API_KEY", None)
            with self.assertRaises(RuntimeError):
                DeepgramSpeech()

    def test_mock_transcribe_media_returns_scripted_phrase(self):
        provider = MockProvider()
        self.assertEqual(provider.transcribe_media(b"", "audio/webm"), provider._stt_script[0])

    def test_override_routes_stt_and_tts_to_speech_backend(self):
        provider = SpeechOverrideProvider(_StubBase(), stt=(dg := _StubSpeech()), tts=dg)
        self.assertEqual(provider.name, "groq+dg-stt+dg-tts")
        self.assertEqual(provider.llm_model, "llm")
        self.assertEqual(provider.chat([]), "chat")            # LLM stays on base
        self.assertEqual(provider.transcribe(b""), "dg-transcribe")
        self.assertEqual(provider.transcribe_media(b""), "dg-media")
        self.assertEqual(provider.synthesize("hi"), b"dg-audio")
        self.assertEqual(provider.stt_model, "nova-3")

    def test_override_stt_only_leaves_tts_on_base(self):
        provider = SpeechOverrideProvider(_StubBase(), stt=_StubSpeech())
        self.assertEqual(provider.name, "groq+dg-stt")  # name reflects only STT routed
        self.assertEqual(provider.transcribe(b""), "dg-transcribe")
        self.assertEqual(provider.synthesize("hi"), b"base-audio")
        self.assertEqual(provider.tts_model, "base-tts")

    def test_voice_for_delegates_to_speech_when_deepgram_tts_active(self):
        provider = SpeechOverrideProvider(_StubBase(), tts=_StubSpeech())
        self.assertEqual(provider.voice_for("es"), "aura-2-celeste-es")  # per-turn native voice
        self.assertEqual(provider.voice_for("en"), "aura-2-luna-en")

    def test_voice_for_reports_base_voice_when_tts_stays_on_base(self):
        # STT-only Deepgram: TTS runs on the base provider, so voice_for must report the
        # base voice (not a Deepgram voice) and tts_model stays the base model -- this is
        # what keeps the browser TTS label correct in a two-vendor stack.
        provider = SpeechOverrideProvider(_StubBase(), stt=_StubSpeech())
        self.assertEqual(provider.voice_for("es"), "base-voice")
        self.assertEqual(provider.tts_model, "base-tts")

    def test_first_transcript_extracts_and_trims(self):
        from types import SimpleNamespace as NS
        resp = NS(results=NS(channels=[NS(alternatives=[NS(transcript="  book a room  ")])]))
        self.assertEqual(_first_transcript(resp), "book a room")

    def test_first_transcript_edge_cases_return_empty(self):
        from types import SimpleNamespace as NS
        self.assertEqual(_first_transcript(NS(results=None)), "")             # failed/accepted job
        self.assertEqual(_first_transcript(NS(results=NS(channels=[]))), "")  # silence
        self.assertEqual(_first_transcript(object()), "")                    # unexpected shape

    def test_override_tts_respects_system_backend(self):
        base = _StubBase()
        base.tts_backend = "system"
        base.synthesize = lambda text, language=None: None  # local `say` returns None
        provider = SpeechOverrideProvider(base, tts=_StubSpeech())
        self.assertIsNone(provider.synthesize("hi"))  # system backend -> not Deepgram

    def test_deepgram_synthesize_picks_native_voice_by_language(self):
        from types import SimpleNamespace as NS
        captured = {}

        class FakeAudio:
            def generate(self, *, text, model, **kw):
                captured["model"] = model
                return iter([b"RIFF", b"data"])

        dg = DeepgramSpeech.__new__(DeepgramSpeech)  # bypass __init__ (no key/SDK)
        dg._voices = {"en": "aura-2-luna-en", "es": "aura-2-celeste-es"}
        dg._client = NS(speak=NS(v1=NS(audio=FakeAudio())))

        self.assertEqual(dg.synthesize("hola", "es"), b"RIFFdata")
        self.assertEqual(captured["model"], "aura-2-celeste-es")  # Spanish -> native ES voice
        dg.synthesize("hi", "en")
        self.assertEqual(captured["model"], "aura-2-luna-en")     # English -> native EN voice
        dg.synthesize("hi", None)
        self.assertEqual(captured["model"], "aura-2-luna-en")     # default -> EN
        # voice_for reports the SAME voice the audio uses (fixes the telemetry label).
        self.assertEqual(dg.voice_for("es"), "aura-2-celeste-es")
        self.assertEqual(dg.voice_for("en"), "aura-2-luna-en")
        self.assertEqual(dg.voice_for(None), "aura-2-luna-en")

    def test_deepgram_transcribe_passes_multi_and_keyterm(self):
        from types import SimpleNamespace as NS
        captured = {}

        class FakeMedia:
            def transcribe_file(self, *, request, model, language, smart_format, keyterm):
                captured.update(model=model, language=language, keyterm=keyterm)
                return NS(results=NS(channels=[NS(alternatives=[NS(transcript="  book a room  ")])]))

        dg = DeepgramSpeech.__new__(DeepgramSpeech)  # bypass __init__ (no key/SDK)
        dg.stt_model = "nova-3"
        dg.stt_language = "multi"
        dg._client = NS(listen=NS(v1=NS(media=FakeMedia())))

        self.assertEqual(dg.transcribe_media(b"audio", "audio/webm"), "book a room")
        self.assertEqual(captured["language"], "multi")   # default: multilingual
        self.assertIn("Vera", captured["keyterm"])        # hotel vocab biasing


class DatePromptTests(unittest.TestCase):
    def test_system_prompt_carries_todays_date(self):
        from datetime import date

        agent = Agent(make_provider("mock"))
        agent.respond("I need a room for two guests",
                      trace=TurnTrace(session_id="t", turn_id="date"))
        sysmsg = agent.messages[0]["content"]
        self.assertIn("Today's date is", sysmsg)          # so the LLM can resolve
        self.assertIn(str(date.today().year), sysmsg)     # "this weekend" -> concrete dates

    def test_prompt_carries_relative_date_resolution_guidance(self):
        # The mock ignores dates, so the only offline guard on the resolution behaviour
        # is the guidance text itself -- assert it survives (weekend rule + self-resolve).
        from agent import _dated_system_prompt

        guidance = _dated_system_prompt()
        self.assertIn("resolve them YOURSELF", guidance)
        self.assertIn("upcoming Saturday", guidance)       # weekend convention is stated
        self.assertIn("starts today", guidance)            # today-is-weekend clause


class _StubStreamClient:
    """Fake OpenAI-dialect client whose chat.completions.create(stream=True)
    yields the given (content, tool_calls) chunks as delta objects."""

    def __init__(self, chunks):
        from types import SimpleNamespace as NS
        self._chunks = chunks
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        from types import SimpleNamespace as NS
        assert kwargs.get("stream") is True  # chat_stream must set stream=True
        return (NS(choices=[NS(delta=NS(content=c, tool_calls=tc))])
                for c, tc in self._chunks)


class StreamingTests(unittest.TestCase):
    def test_sentence_buffer_emits_on_terminators_and_flushes_tail(self):
        buf = SentenceBuffer()
        self.assertEqual(buf.feed("Hello there"), [])
        self.assertEqual(buf.feed("! How can I help you"), ["Hello there!"])
        self.assertEqual(buf.feed("? "), ["How can I help you?"])
        self.assertEqual(buf.feed("Room is $3.50 a night"), [])  # decimal not split
        self.assertEqual(buf.flush(), "Room is $3.50 a night")
        self.assertEqual(buf.flush(), "")  # buffer cleared after flush

    def test_sentence_buffer_splits_multiple_in_one_delta(self):
        buf = SentenceBuffer()
        self.assertEqual(
            buf.feed("One. Two! Three? four"),
            ["One.", "Two!", "Three?"],
        )
        self.assertEqual(buf.flush(), "four")

    def test_streaming_enabled_requires_flag_and_capability(self):
        class Capable:
            tts_backend = "provider"

            def chat_stream(self, *a, **k):
                return None

        cap = Capable()
        with patch.dict(os.environ, {"STREAMING": "off"}):
            self.assertFalse(streaming_enabled(cap))            # flag off -> batch
        with patch.dict(os.environ, {"STREAMING": "on"}):
            self.assertTrue(streaming_enabled(cap))             # flag on + capable
            self.assertFalse(streaming_enabled(cap, text_mode=True))   # text -> batch
            self.assertFalse(streaming_enabled(make_provider("mock")))  # no chat_stream
            cap.tts_backend = "system"
            self.assertFalse(streaming_enabled(cap))            # local TTS -> batch

    def test_chat_stream_streams_text_and_rebuilds_message(self):
        provider = Provider.__new__(Provider)  # bypass __init__ (no key/SDK)
        provider.llm_model = "gpt-4.1-mini"
        provider.client = _StubStreamClient([("Hello", None), (" there.", None)])
        deltas = []
        resp = provider.chat_stream([{"role": "user", "content": "hi"}],
                                    tools=[{"type": "function"}], on_delta=deltas.append)
        msg = resp.choices[0].message
        self.assertEqual(deltas, ["Hello", " there."])          # streamed in order
        self.assertEqual(msg.content, "Hello there.")           # reassembled == batch
        self.assertIsNone(msg.tool_calls)

    def test_chat_stream_reassembles_tool_call_fragments(self):
        from types import SimpleNamespace as NS
        provider = Provider.__new__(Provider)
        provider.llm_model = "gpt-4.1-mini"
        frag_a = NS(index=0, id="call_1",
                    function=NS(name="check_availability", arguments='{"gu'))
        frag_b = NS(index=0, id=None, function=NS(name=None, arguments='ests": 2}'))
        provider.client = _StubStreamClient([(None, [frag_a]), (None, [frag_b])])
        fired = []
        resp = provider.chat_stream([], tools=[{"type": "function"}], on_delta=fired.append)
        msg = resp.choices[0].message
        self.assertEqual(fired, [])                              # no content -> no on_delta
        self.assertIsNone(msg.content)
        self.assertEqual(len(msg.tool_calls), 1)
        call = msg.tool_calls[0]
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.function.name, "check_availability")
        self.assertEqual(call.function.arguments, '{"guests": 2}')  # fragments joined

    def test_speech_override_chat_stream_delegates_to_base(self):
        provider = SpeechOverrideProvider(_StubBase(), stt=_StubSpeech())
        fired = []
        resp = provider.chat_stream([], on_delta=fired.append)
        self.assertEqual(fired, ["chat"])                       # base's on_delta ran
        self.assertEqual(resp.choices[0].message.content, "chat")

    def test_agent_respond_streams_deltas_with_reply_unchanged(self):
        from types import SimpleNamespace as NS

        class FakeStreamProvider:
            name = "fake"
            llm_model = "fake-llm"
            tts_backend = "provider"

            def chat(self, messages, tools=None, tool_choice=None):
                raise AssertionError("batch chat() must not run when streaming")

            def chat_stream(self, messages, tools=None, tool_choice=None, on_delta=None):
                for piece in ("Sure", ", ", "booking now."):
                    if on_delta:
                        on_delta(piece)
                return NS(choices=[NS(message=NS(content="Sure, booking now.", tool_calls=None))])

        agent = Agent(FakeStreamProvider())
        deltas = []
        reply, action = agent.respond("book a room",
                                      trace=TurnTrace(session_id="s"), on_delta=deltas.append)
        self.assertEqual("".join(deltas), reply)                # streamed text == reply
        self.assertEqual(reply, "Sure, booking now.")
        self.assertIsNone(action)

    def test_mock_ignores_on_delta_and_matches_batch(self):
        reset_inventory()
        batch_reply, batch_action = Agent(make_provider("mock")).respond(
            "I need a room for two guests", trace=TurnTrace(session_id="s"))
        reset_inventory()
        deltas = []
        stream_reply, stream_action = Agent(make_provider("mock")).respond(
            "I need a room for two guests", trace=TurnTrace(session_id="s"),
            on_delta=deltas.append)
        self.assertEqual(batch_reply, stream_reply)             # identical reply
        self.assertEqual(batch_action, stream_action)
        self.assertEqual(deltas, [])                            # mock never streams

    def test_tool_preamble_is_not_fused_into_reply(self):
        # A model that narrates a preamble alongside a tool call must NOT have that
        # preamble concatenated (spaceless) into the final reply. on_reset drops it.
        import voice_loop
        from types import SimpleNamespace as NS

        class PreambleProvider:
            name = "fake"; llm_model = "fake"; tts_backend = "provider"

            def __init__(self):
                self.calls = 0

            def chat(self, *a, **k):
                raise AssertionError("batch chat() must not run when streaming")

            def chat_stream(self, messages, tools=None, tool_choice=None, on_delta=None):
                self.calls += 1
                if self.calls == 1:  # preamble (no terminator) alongside a tool call
                    if on_delta:
                        on_delta("One moment while I check")
                    tc = NS(id="c1", type="function", function=NS(
                        name="check_availability",
                        arguments='{"check_in":"August 12","check_out":"August 14","guests":2}'))
                    return NS(choices=[NS(message=NS(content="One moment while I check", tool_calls=[tc]))])
                if on_delta:  # the actual reply
                    on_delta("The Standard Queen is available.")
                return NS(choices=[NS(message=NS(content="The Standard Queen is available.", tool_calls=None))])

        spoken: list[str] = []
        original = voice_loop._play_sentence
        voice_loop._play_sentence = lambda provider, sentence, language: spoken.append(sentence)
        try:
            reset_inventory()
            provider = PreambleProvider()
            reply, _ = voice_loop._respond_streaming(
                provider, Agent(provider), "book a room", TurnTrace(session_id="s"))
        finally:
            voice_loop._play_sentence = original

        self.assertEqual(reply, "The Standard Queen is available.")
        self.assertEqual(spoken, ["The Standard Queen is available."])   # preamble dropped
        self.assertNotIn("checkThe", "".join(spoken))                    # no spaceless fusion

    def test_mid_stream_error_still_speaks_the_degrade_reply(self):
        # If the stream drops after partial content, the caller must still hear the
        # transfer apology (not just an orphaned fragment). on_reset discards the partial.
        import voice_loop

        class DroppingProvider:
            name = "fake"; llm_model = "fake"; tts_backend = "provider"

            def chat(self, *a, **k):
                raise AssertionError("batch chat() must not run when streaming")

            def chat_stream(self, messages, tools=None, tool_choice=None, on_delta=None):
                if on_delta:
                    on_delta("The room ")          # partial, no terminator...
                raise RuntimeError("connection dropped")  # ...then the stream fails

        spoken: list[str] = []
        original = voice_loop._play_sentence
        voice_loop._play_sentence = lambda provider, sentence, language: spoken.append(sentence)
        try:
            provider = DroppingProvider()
            reply, action = voice_loop._respond_streaming(
                provider, Agent(provider), "any room?", TurnTrace(session_id="s"))
        finally:
            voice_loop._play_sentence = original

        self.assertEqual(action, "transfer")           # degraded gracefully
        self.assertTrue(reply.strip())                 # an apology was returned
        self.assertEqual(spoken, [reply])              # ...and it was actually spoken
        self.assertNotIn("The room", spoken)           # orphaned partial was dropped


class _FakeVendorProvider:
    """Stands in for Provider in make_provider matrix tests (no keys, no SDK)."""

    constructed: list[str] = []

    def __init__(self, name=None):
        import os as _os
        name = (name or _os.getenv("PROVIDER", "groq")).lower()
        type(self).constructed.append(name)
        self.name = name
        self.llm_model = f"{name}-llm"
        self.stt_model = f"{name}-stt"
        self.tts_model = f"{name}-tts"
        self.tts_voice = f"{name}-voice"
        self.tts_backend = "provider"

    def transcribe(self, pcm, sample_rate=16000):
        return f"{self.name}-transcribe"

    def synthesize(self, text, language=None):
        return f"{self.name}-audio".encode()


class _FakeDeepgram:
    name = "deepgram"
    constructed: list[str] = []

    def __init__(self):
        type(self).constructed.append("deepgram")
        self.stt_model = "nova-3"
        self.tts_model = "aura-2-luna-en"
        self.tts_voice = "aura-2-luna-en"

    def synthesize(self, text, language=None):
        return b"dg-audio"


class ProviderMixTests(unittest.TestCase):
    """make_provider matrix: STT_PROVIDER/TTS_PROVIDER accept openai|groq|deepgram."""

    def setUp(self):
        _FakeVendorProvider.constructed = []
        _FakeDeepgram.constructed = []
        patcher_p = patch("providers.Provider", _FakeVendorProvider)
        patcher_d = patch("deepgram_speech.DeepgramSpeech", _FakeDeepgram)
        patcher_p.start(); patcher_d.start()
        self.addCleanup(patcher_p.stop)
        self.addCleanup(patcher_d.stop)

    def test_unknown_override_lists_all_vendors_before_any_key(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "whisper-cloud"}):
            with self.assertRaises(ValueError) as ctx:
                make_provider("openai")
        for vendor in ("deepgram", "groq", "openai"):
            self.assertIn(vendor, str(ctx.exception))
        self.assertEqual(_FakeVendorProvider.constructed, [])  # fail-fast, key-free

    def test_stt_provider_openai_builds_second_provider(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "openai", "TTS_PROVIDER": ""}):
            provider = make_provider("groq")
        self.assertEqual(provider.name, "groq+oai-stt")
        self.assertEqual(provider.transcribe(b""), "openai-transcribe")  # STT routed
        self.assertEqual(provider.synthesize("hi"), b"groq-audio")       # TTS on base
        self.assertEqual(provider.stt_model, "openai-stt")

    def test_tts_provider_groq_under_openai_base(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "", "TTS_PROVIDER": "groq", "TTS_BACKEND": "provider"}):
            provider = make_provider("openai")
        self.assertEqual(provider.name, "openai+groq-tts")
        self.assertEqual(provider.synthesize("hi"), b"groq-audio")
        self.assertEqual(provider.transcribe(b""), "openai-transcribe")

    def test_override_matching_base_is_noop(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "groq", "TTS_PROVIDER": ""}):
            provider = make_provider("groq")
        self.assertIsInstance(provider, _FakeVendorProvider)  # bare base, no wrapper
        self.assertEqual(provider.name, "groq")

    def test_same_override_vendor_constructed_once_across_stages(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "openai", "TTS_PROVIDER": "openai", "TTS_BACKEND": "provider"}):
            provider = make_provider("groq")
        self.assertEqual(provider.name, "groq+oai-stt+oai-tts")
        # base groq + ONE shared openai backend
        self.assertEqual(_FakeVendorProvider.constructed, ["groq", "openai"])

    def test_three_vendor_mix(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "openai", "TTS_PROVIDER": "deepgram", "TTS_BACKEND": "provider"}):
            provider = make_provider("groq")
        self.assertEqual(provider.name, "groq+oai-stt+dg-tts")
        self.assertEqual(provider.transcribe(b""), "openai-transcribe")
        self.assertEqual(provider.synthesize("hi"), b"dg-audio")
        self.assertEqual(_FakeDeepgram.constructed, ["deepgram"])

    def test_voice_for_falls_back_to_static_voice_for_plain_provider_tts(self):
        # A plain Provider TTS backend has no voice_for -> static tts_voice label.
        with patch.dict(os.environ, {"STT_PROVIDER": "", "TTS_PROVIDER": "openai", "TTS_BACKEND": "provider"}):
            provider = make_provider("groq")
        self.assertEqual(provider.voice_for("es"), "openai-voice")

    def test_mock_still_ignores_all_overrides(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "openai", "TTS_PROVIDER": "deepgram", "TTS_BACKEND": "provider"}):
            provider = make_provider("mock")
        self.assertIsInstance(provider, MockProvider)
        self.assertEqual(_FakeVendorProvider.constructed, [])

    def test_system_tts_backend_skips_tts_override_construction(self):
        # TTS_BACKEND=system means the override would never be used -- it must not
        # be constructed (no key demanded) and the provider stays unwrapped.
        with patch.dict(os.environ, {"STT_PROVIDER": "", "TTS_PROVIDER": "groq",
                                     "TTS_BACKEND": "system"}):
            provider = make_provider("openai")
        self.assertIsInstance(provider, _FakeVendorProvider)  # bare base
        self.assertEqual(_FakeVendorProvider.constructed, ["openai"])  # no groq built


class DeepgramHardeningTests(unittest.TestCase):
    def _dg_with_fake_speak(self, behaviors):
        """DeepgramSpeech via __new__ with a generate() scripted per call."""
        from types import SimpleNamespace as NS
        calls = {"n": 0}

        def generate(**kwargs):
            behavior = behaviors[min(calls["n"], len(behaviors) - 1)]
            calls["n"] += 1
            if isinstance(behavior, Exception):
                raise behavior
            return iter(behavior)

        dg = DeepgramSpeech.__new__(DeepgramSpeech)
        dg._voices = {"en": "aura-2-luna-en", "es": "aura-2-celeste-es"}
        dg._client = NS(speak=NS(v1=NS(audio=NS(generate=generate))))
        return dg, calls

    def test_synthesize_retries_once_on_dropped_connection(self):
        dg, calls = self._dg_with_fake_speak([ConnectionError("dropped"), [b"RIFF", b"ok"]])
        self.assertEqual(dg.synthesize("hi", "en"), b"RIFFok")
        self.assertEqual(calls["n"], 2)  # failed once, retried once

    def test_synthesize_gives_up_after_one_retry(self):
        dg, calls = self._dg_with_fake_speak([ConnectionError("down")])
        with self.assertRaises(ConnectionError):
            dg.synthesize("hi", "en")
        self.assertEqual(calls["n"], 2)  # exactly two attempts, then propagate

    def test_api_errors_are_not_retried(self):
        dg, calls = self._dg_with_fake_speak([ValueError("bad voice")])
        with self.assertRaises(ValueError):
            dg.synthesize("hi", "en")
        self.assertEqual(calls["n"], 1)  # non-transient -> no retry

    def test_stt_language_env_knob(self):
        from types import SimpleNamespace as NS
        captured = {}

        class FakeMedia:
            def transcribe_file(self, *, request, model, language, smart_format, keyterm):
                captured["language"] = language
                return NS(results=NS(channels=[NS(alternatives=[NS(transcript="ok")])]))

        dg = DeepgramSpeech.__new__(DeepgramSpeech)
        dg.stt_model = "nova-3"
        dg.stt_language = "en"  # what STT_LANGUAGE=en produces
        dg._client = NS(listen=NS(v1=NS(media=FakeMedia())))
        self.assertEqual(dg.transcribe_media(b"audio"), "ok")
        self.assertEqual(captured["language"], "en")

    def test_generic_voice_and_language_env_knobs(self):
        # The Deepgram backend reads GENERIC knobs -- TTS_VOICE (EN), TTS_VOICE_ES (ES),
        # STT_LANGUAGE -- not vendor-prefixed ones. Guards against a regression to the
        # old DEEPGRAM_TTS_VOICE_ES / DEEPGRAM_STT_LANGUAGE names.
        import sys
        import types

        fake = types.ModuleType("deepgram")
        fake.DeepgramClient = lambda api_key: object()
        with patch.dict(sys.modules, {"deepgram": fake}), patch.dict(
            os.environ,
            {"DEEPGRAM_API_KEY": "test-key", "TTS_VOICE": "aura-2-thalia-en",
             "TTS_VOICE_ES": "aura-2-diana-es", "STT_LANGUAGE": "en"},
        ):
            dg = DeepgramSpeech()
        self.assertEqual(dg.voice_for("en"), "aura-2-thalia-en")
        self.assertEqual(dg.voice_for("es"), "aura-2-diana-es")
        self.assertEqual(dg.stt_language, "en")

    def test_non_deepgram_stt_model_fails_fast_at_boot(self):
        # A stale non-Deepgram STT_MODEL (an OpenAI/Groq value left set when routing STT
        # to Deepgram) must raise at construction, not crash on the first transcription.
        import sys
        import types

        fake = types.ModuleType("deepgram")
        fake.DeepgramClient = lambda api_key: object()
        with patch.dict(sys.modules, {"deepgram": fake}), patch.dict(
            os.environ, {"DEEPGRAM_API_KEY": "k", "STT_MODEL": "whisper-large-v3-turbo"},
        ):
            with self.assertRaises(RuntimeError):
                DeepgramSpeech()

    def test_non_deepgram_tts_voice_fails_fast_at_boot(self):
        # TTS_VOICE=alloy (OpenAI) leaking into a Deepgram-owned TTS stage must raise at
        # construction rather than sending "alloy" as a voice and 400-ing on first synth.
        import sys
        import types

        fake = types.ModuleType("deepgram")
        fake.DeepgramClient = lambda api_key: object()
        with patch.dict(sys.modules, {"deepgram": fake}), patch.dict(
            os.environ, {"DEEPGRAM_API_KEY": "k", "TTS_VOICE": "alloy"},
        ):
            with self.assertRaises(RuntimeError):
                DeepgramSpeech()


if __name__ == "__main__":
    unittest.main()
