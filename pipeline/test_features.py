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
    SpeechOverrideProvider,
    _env_or_default,
    _mk_tool,
    is_noise_transcript,
    is_stt_hallucination,
    make_provider,
)
from router import AgentRouter
from scale_check import estimate_capacity
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

    def transcribe(self, pcm, sample_rate=16000):
        return "base-transcribe"

    def transcribe_media(self, audio, content_type=""):
        return "base-media"

    def synthesize(self, text, language=None):
        return b"base-audio"


class _StubSpeech:
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
        provider = SpeechOverrideProvider(_StubBase(), _StubSpeech(), use_stt=True, use_tts=True)
        self.assertEqual(provider.name, "groq+dg-stt+dg-tts")
        self.assertEqual(provider.llm_model, "llm")
        self.assertEqual(provider.chat([]), "chat")            # LLM stays on base
        self.assertEqual(provider.transcribe(b""), "dg-transcribe")
        self.assertEqual(provider.transcribe_media(b""), "dg-media")
        self.assertEqual(provider.synthesize("hi"), b"dg-audio")
        self.assertEqual(provider.stt_model, "nova-3")

    def test_override_stt_only_leaves_tts_on_base(self):
        provider = SpeechOverrideProvider(_StubBase(), _StubSpeech(), use_stt=True, use_tts=False)
        self.assertEqual(provider.name, "groq+dg-stt")  # name reflects only STT routed
        self.assertEqual(provider.transcribe(b""), "dg-transcribe")
        self.assertEqual(provider.synthesize("hi"), b"base-audio")
        self.assertEqual(provider.tts_model, "base-tts")

    def test_voice_for_delegates_to_speech_when_deepgram_tts_active(self):
        provider = SpeechOverrideProvider(_StubBase(), _StubSpeech(), use_stt=False, use_tts=True)
        self.assertEqual(provider.voice_for("es"), "aura-2-celeste-es")  # per-turn native voice
        self.assertEqual(provider.voice_for("en"), "aura-2-luna-en")

    def test_voice_for_reports_base_voice_when_tts_stays_on_base(self):
        # STT-only Deepgram: TTS runs on the base provider, so voice_for must report the
        # base voice (not a Deepgram voice) and tts_model stays the base model -- this is
        # what keeps the browser TTS label correct in a two-vendor stack.
        provider = SpeechOverrideProvider(_StubBase(), _StubSpeech(), use_stt=True, use_tts=False)
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
        provider = SpeechOverrideProvider(base, _StubSpeech(), use_stt=False, use_tts=True)
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
        dg._client = NS(listen=NS(v1=NS(media=FakeMedia())))

        self.assertEqual(dg.transcribe_media(b"audio", "audio/webm"), "book a room")
        self.assertEqual(captured["language"], "multi")   # always multilingual
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


if __name__ == "__main__":
    unittest.main()
