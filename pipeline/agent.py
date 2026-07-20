"""
agent.py  -  the "brain" (Layer B). LLM + tool loop over a Provider.

Tools mirror a hotel reservations desk:
    check_availability -> find matching rooms
    create_booking     -> reserve a room
    transfer_to_human  -> front desk / human queue
    end_call           -> caller done (real system: SIP BYE)

Uses OpenAI-style function calling, which both Groq and OpenAI support, so this
file is provider-agnostic  -  it only talks to Provider.chat().
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

from inventory import book_if_available, is_available, parse_date
from knowledge import search_hotel_knowledge
from providers import Provider
from router import AgentRouter, LANGUAGES
from telemetry import TurnTrace

# Safety cap on the tool-calling loop. A model that keeps emitting tool calls
# (or a provider that errors) must never spin forever or dead-end the caller.
# Real multi-tool turns settle in 2-3 iterations; 6 is generous headroom.
# Env-overridable for load testing.
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "6"))

SYSTEM_PROMPT = """You are a friendly phone reservations agent for Vera Hotel.
Your only job is hotel room booking support: new reservations, availability,
room options, rates returned by tools, changing/canceling reservations, and
transferring to the front desk. Hotel policies and amenities are in scope even
when the caller asks about them during an incomplete booking flow.

Guardrails:
- Do not answer questions outside hotel booking support, including weather,
  news, trivia, coding, medical, legal, finance, or general assistant tasks.
- For off-topic requests, politely say you can only help with hotel reservations
  and ask whether they want to book, change, or cancel a stay.
- Never invent availability, rates, confirmation numbers, policies, or guest
  details. Use tools for availability and booking. Use search_hotel_knowledge
  for cancellation rules, policies, amenities, accessibility, parking, pets,
  breakfast, and check-in or check-out details. Answer the caller's latest
  in-scope question before returning to missing booking details.
- Keep replies short and spoken-friendly: one or two sentences, no bullet lists,
  no markdown, no emoji.
- When the caller asks to speak, continue, switch, or switch back in a supported
  language, call set_language immediately. Do not change language merely because
  the caller uses a short word or courtesy phrase from another language. After
  the tool result, answer in the selected language.

Booking flow:
1. First collect only check-in date, check-out date, guest count, and optional
   room type preference.
2. Once dates and guests are known, call check_availability immediately, even
   if no room type preference was given.
3. Offer the available room options and ask which one they want.
4. Only after the caller chooses or confirms a room, collect guest name and
   phone or email.
5. Before booking, summarize the selected room and ask for confirmation.
6. After the caller confirms and required details are present, call create_booking.
7. If the caller asks for a person or the request is outside what you can do,
   call transfer_to_human. When the conversation is clearly over, call end_call."""

# OpenAI-style tool schema (works on Groq too).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_language",
            "description": "Set the response language for this call when the caller asks to speak, "
                           "continue, switch, or switch back in English or Spanish. Only call for an "
                           "explicit language-change request, not an isolated foreign word or courtesy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["en", "es"],
                        "description": "Requested response language: en for English or es for Spanish.",
                    },
                },
                "required": ["language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check hotel room availability for dates, guests, and optional room type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "check_in": {
                        "type": "string",
                        "description": "Check-in date as stated by the caller.",
                    },
                    "check_out": {
                        "type": "string",
                        "description": "Check-out date as stated by the caller.",
                    },
                    "guests": {
                        "type": "integer",
                        "description": "Number of guests.",
                    },
                    "room_type": {
                        "type": "string",
                        "description": "Optional preference: standard, king, suite, family, or accessible.",
                    },
                },
                "required": ["check_in", "check_out", "guests"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Create a hotel booking after the caller confirms the room option.",
            "parameters": {
                "type": "object",
                "properties": {
                    "check_in": {"type": "string"},
                    "check_out": {"type": "string"},
                    "guests": {"type": "integer"},
                    "room_type": {"type": "string"},
                    "guest_name": {"type": "string"},
                    "contact": {
                        "type": "string",
                        "description": "Phone number or email for the booking.",
                    },
                },
                "required": [
                    "check_in",
                    "check_out",
                    "guests",
                    "room_type",
                    "guest_name",
                    "contact",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotel_knowledge",
            "description": "Retrieve grounded Vera Hotel policies, amenities, and operating details. "
                           "Always use for cancellation rules, check-in or check-out times, parking, "
                           "pets, breakfast, accessibility, and other hotel-information questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The caller's policy or hotel-information question.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Hand the call to a human agent queue. Use when the caller "
                           "asks for a person or the request is out of scope.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call politely when the conversation is finished.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_KNOWLEDGE_INTENT_PHRASES = (
    "cancellation policy", "cancelation policy", "cancellation fee", "cancel fee",
    "cancellation charge", "when can i cancel", "refundable", "non-refundable",
    "pet policy", "pets allowed", "dogs allowed", "bring my dog", "bring a pet",
    "parking", "valet", "breakfast", "check-in", "check in", "check-out",
    "check out", "accessibility", "accessible room", "wi-fi", "wifi", "amenities",
    "política de cancelación", "politica de cancelacion", "mascotas",
    "estacionamiento", "desayuno", "accesibilidad",
)

_FUZZY_AMENITY_TERMS = (
    "mascota", "mascotas", "pet", "pets", "parking", "estacionamiento",
    "breakfast", "desayuno", "accessibility", "accesibilidad", "wifi",
)

_LANGUAGE_NAMES = {
    "en": {"english", "ingles"},
    "es": {"spanish", "espanol"},
}


def _normalized_tokens(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", normalized)


def _has_fuzzy_term(tokens: list[str], terms: tuple[str, ...], cutoff: float = 0.82) -> bool:
    return any(
        SequenceMatcher(None, token, term).ratio() >= cutoff
        for token in tokens
        for term in terms
    )


def explicit_language_request(text: str, language: str) -> bool:
    """Require the target language name before allowing a session-state change."""
    return bool(set(_normalized_tokens(text)) & _LANGUAGE_NAMES.get(language, set()))


def required_tool_for(text: str) -> str | None:
    """Route high-confidence knowledge intents before probabilistic LLM selection."""
    normalized = " ".join(text.lower().split())
    if any(phrase in normalized for phrase in _KNOWLEDGE_INTENT_PHRASES):
        return "search_hotel_knowledge"
    tokens = _normalized_tokens(text)
    if _has_fuzzy_term(tokens, _FUZZY_AMENITY_TERMS):
        return "search_hotel_knowledge"
    has_policy = _has_fuzzy_term(tokens, ("policy", "politica"))
    has_cancellation = _has_fuzzy_term(tokens, ("cancellation", "cancelacion"))
    if has_policy and has_cancellation:
        return "search_hotel_knowledge"
    return None


def _named_tool_choice(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


# --- Mock tool implementations (swap for real backends in production) ---

_ROOMS = {
    "standard": {"name": "Standard Queen", "rate": "$189/night", "capacity": 2},
    "king": {"name": "Deluxe King", "rate": "$229/night", "capacity": 2},
    "suite": {"name": "Harbor Suite", "rate": "$329/night", "capacity": 4},
    "family": {"name": "Family Double Queen", "rate": "$269/night", "capacity": 5},
    "accessible": {"name": "Accessible Queen", "rate": "$199/night", "capacity": 2},
}


def _normalize_room_type(value: str | None) -> str | None:
    room_type = (value or "").strip().lower()
    if not room_type:
        return None
    for key in _ROOMS:
        if key in room_type:
            return key
    if "double" in room_type:
        return "family"
    if "queen" in room_type:
        return "standard"
    return None


def _parse_guests(args: dict) -> int | None:
    """Guest count: default 1 when absent, or None when present but not a whole
    number (so _validate_stay can reject it rather than crashing on int())."""
    raw = args.get("guests")
    if raw is None or raw == "":
        return 1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _validate_stay(args: dict, guests: int | None) -> str | None:
    """Return a spoken-friendly clarification if the stay is invalid, else None.

    Guards the booking tools against bad input (non-numeric/zero guests,
    unparseable/reversed dates) instead of silently proceeding. The live LLM works
    the clarification into its reply; the offline mock speaks it verbatim.
    """
    if guests is None or guests < 1:
        return ("I need a valid number of guests for the stay. "
                "How many people will be staying?")
    check_in = parse_date(args.get("check_in"))
    check_out = parse_date(args.get("check_out"))
    if check_in is None or check_out is None:
        return ("I need a valid check-in and check-out date. "
                "Could you tell me those dates again?")
    if check_out <= check_in:
        # A genuine new-year wrap (Dec 30 -> Jan 2) is a SHORT stay once check-out
        # rolls to next year; a same-year reversed range (Aug 14 -> Aug 12) rolls to
        # ~363 days and is a mistake. Accept only the short wrap; reject the rest.
        rolled = check_out.replace(year=check_out.year + 1)
        if not 0 < (rolled - check_in).days <= 30:
            return ("The check-out date needs to be after the check-in date. "
                    "Could you confirm the dates?")
    return None


def run_tool(name: str, args: dict) -> dict:
    """Execute a tool call. The optional 'action' key is a control signal for
    the voice loop ('transfer' -> SIP REFER, 'hangup' -> SIP BYE)."""
    if name == "check_availability":
        guests = _parse_guests(args)
        invalid = _validate_stay(args, guests)
        if invalid:
            return {"result": invalid}
        preferred = _normalize_room_type(args.get("room_type"))
        check_in = args.get("check_in")
        check_out = args.get("check_out")
        rooms = []
        for key, room in _ROOMS.items():
            if preferred and key != preferred:
                continue
            if guests > room["capacity"]:
                continue
            if not is_available(key, check_in, check_out):
                continue  # already booked for overlapping dates
            rooms.append(f"{room['name']} at {room['rate']}")
        if not rooms:
            return {
                "result": "No matching rooms are available for those dates and guest count. "
                          "Offer to transfer to the front desk.",
            }
        return {
            "result": "Available rooms for "
                      f"{check_in} to {check_out}: "
                      f"{'; '.join(rooms)}.",
        }
    if name == "create_booking":
        guests = _parse_guests(args)
        invalid = _validate_stay(args, guests)
        if invalid:
            return {"result": invalid}
        room_key = _normalize_room_type(args.get("room_type")) or "standard"
        room = _ROOMS[room_key]
        if guests > room["capacity"]:  # match check_availability's capacity rule
            return {
                "result": f"The {room['name']} holds up to {room['capacity']} guests. "
                          "Offer a larger room or transfer to the front desk.",
            }
        check_in = args.get("check_in")
        check_out = args.get("check_out")
        # Atomic: the availability check and the reservation happen together, so
        # two concurrent callers can't both book the same room+dates.
        confirmation = book_if_available(
            room_key, check_in, check_out,
            guest_name=args.get("guest_name"),
            contact=args.get("contact"),
        )
        if confirmation is None:
            return {
                "result": f"I'm sorry, the {room['name']} was just booked for those dates. "
                          "I can check other rooms or connect you with the front desk.",
            }
        return {
            "result": f"Booking confirmed. Confirmation {confirmation} for "
                      f"{args.get('guest_name')} in a {room['name']} from "
                      f"{check_in} to {check_out} for "
                      f"{guests} guest(s). Confirmation sent to "
                      f"{args.get('contact')}.",
        }
    if name == "search_hotel_knowledge":
        return search_hotel_knowledge(str(args.get("query", "")))
    if name == "transfer_to_human":
        return {"result": "Transferring you to the front desk.", "action": "transfer"}
    if name == "end_call":
        return {"result": "Ending the call.", "action": "hangup"}
    return {"result": f"Unknown tool: {name}"}


class Agent:
    """LLM + tool loop for one call. Holds conversation history."""

    def __init__(self, provider: Provider):
        self.provider = provider
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.router = AgentRouter()
        self.current_language = "en"
        self.current_locale = LANGUAGES["en"]["locale"]
        self.last_trace: TurnTrace | None = None
        self.last_sources: list[str] = []

    def respond(self, user_text: str, trace: TurnTrace | None = None) -> tuple[str, str | None]:
        """Take the caller's transcript, return (spoken_reply, action|None).

        Loops until the model produces a plain text reply, executing any tool
        calls in between. `action` is the last control signal seen (transfer/
        hangup), which the voice loop uses to end the call.
        """
        trace = trace or TurnTrace()
        self.last_trace = trace
        self.last_sources = []

        with trace.span("routing"):
            route = self.router.route()
            self.current_language = route.language
            self.current_locale = route.locale
            self.messages[0]["content"] = f"{SYSTEM_PROMPT}\n\n{self.router.instruction()}"
        trace.event(
            "router.selected",
            language=route.language,
            locale=route.locale,
            changed=route.changed,
            reason=route.reason,
        )
        trace.attributes.update({
            "language": route.language,
            "locale": route.locale,
            "provider": getattr(self.provider, "name", "unknown"),
            "model": getattr(self.provider, "llm_model", "unknown"),
        })
        trace.event("caller.transcript", text=user_text)
        self.messages.append({"role": "user", "content": user_text})
        action: str | None = None
        required_tool = required_tool_for(user_text)
        if required_tool:
            trace.event(
                "tool.route_selected",
                tool=required_tool,
                reason="hotel_knowledge_intent",
            )
        first_model_call = True

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                with trace.span("llm", model=getattr(self.provider, "llm_model", "unknown")):
                    tool_choice = (
                        _named_tool_choice(required_tool)
                        if first_model_call and required_tool
                        else None
                    )
                    resp = self.provider.chat(
                        self.messages,
                        tools=TOOLS,
                        tool_choice=tool_choice,
                    )
                    first_model_call = False
                    # Inside the try so a structurally-malformed response (empty
                    # choices / missing message) degrades like an exception does.
                    msg = resp.choices[0].message
            except Exception as exc:
                # Live-provider outage, rate limit, timeout, malformed response:
                # apologize and hand off rather than crash the turn loop.
                trace.event(
                    "provider.error",
                    errorType=type(exc).__name__,
                    message=str(exc),
                )
                return self._degrade_to_transfer(trace, reason="provider_error")

            if not msg.tool_calls:
                reply = msg.content or ""
                self.messages.append({"role": "assistant", "content": reply})
                trace.event("assistant.response", text=reply, action=action)
                return reply, action

            # Record the assistant's tool-call turn, then answer each call.
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                trace.event("tool.requested", tool=tc.function.name, arguments=args)
                with trace.span("tools", tool=tc.function.name):
                    if tc.function.name == "set_language":
                        language = str(args.get("language", "")).lower()
                        try:
                            if not explicit_language_request(user_text, language):
                                trace.event(
                                    "router.language_change_rejected",
                                    requestedLanguage=language,
                                    reason="no_explicit_language_name",
                                )
                                raise PermissionError
                            language_route = self.router.set_language(language)
                            self.current_language = language_route.language
                            self.current_locale = language_route.locale
                            self.messages[0]["content"] = (
                                f"{SYSTEM_PROMPT}\n\n{self.router.instruction()}"
                            )
                            trace.attributes.update({
                                "language": language_route.language,
                                "locale": language_route.locale,
                            })
                            trace.event(
                                "router.language_changed",
                                language=language_route.language,
                                locale=language_route.locale,
                                changed=language_route.changed,
                                reason=language_route.reason,
                            )
                            result = {
                                "result": (
                                    "Response language set to "
                                    f"{LANGUAGES[language_route.language]['name']}."
                                ),
                            }
                        except PermissionError:
                            result = {
                                "result": (
                                    "Language unchanged because the caller did not explicitly "
                                    "request the target language. Continue in the current language."
                                ),
                            }
                        except ValueError:
                            result = {
                                "result": "Unsupported language. Continue in the current language.",
                            }
                    elif tc.function.name == "search_hotel_knowledge":
                        with trace.span("retrieval", query=args.get("query", "")):
                            result = run_tool(tc.function.name, args)
                    else:
                        result = run_tool(tc.function.name, args)
                trace.event(
                    "tool.result",
                    tool=tc.function.name,
                    result=result.get("result", ""),
                    sources=result.get("sources", []),
                    action=result.get("action"),
                )
                self.last_sources.extend(result.get("sources", []))
                if result.get("action"):
                    action = result["action"]
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result["result"],
                })
            # loop again so the model can speak given the tool results

        # Spent the whole tool-iteration budget without the model settling on a
        # spoken reply (a stuck tool loop). Hand off instead of dead-ending.
        trace.event("tool_loop.exhausted", limit=MAX_TOOL_ITERATIONS)
        return self._degrade_to_transfer(trace, reason="max_tool_iterations")

    def _degrade_to_transfer(self, trace: TurnTrace, reason: str) -> tuple[str, str]:
        """Graceful fallback when the model/provider fails or loops.

        Rather than crash or dead-end the caller, apologize and hand off to the
        front desk (SIP REFER). Language-aware so the closing matches the call.
        """
        trace.event("agent.degraded", reason=reason)
        if self.current_language == "es":
            reply = (
                "Lo siento, tengo un problema técnico en este momento. "
                "Le comunico con la recepción."
            )
        else:
            reply = (
                "I'm sorry, I'm having trouble on my end right now. "
                "Let me connect you with the front desk."
            )
        self.messages.append({"role": "assistant", "content": reply})
        trace.event("assistant.response", text=reply, action="transfer")
        return reply, "transfer"
