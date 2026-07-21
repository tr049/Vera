"""Serve a tiny browser client for testing local LiveKit audio.

Run this after `./start_local_server.sh`, then open http://localhost:5173.
"""

from __future__ import annotations

import base64
import json
import os
import string
import sys
import threading
import warnings
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import jwt
from livekit import api

from env_loader import load_env_files

HOST = os.getenv("TALK_HOST", "localhost")
PORT = int(os.getenv("TALK_PORT", "5173"))
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PIPELINE_ROOT = REPO_ROOT / "pipeline"

_session_registry_lock = threading.Lock()
# LRU-bounded so a flood of distinct X-Session-IDs can't grow the registry without
# limit (each id builds a full Agent). Ordered so the oldest idle session evicts first.
_agent_sessions: "OrderedDict[str, object]" = OrderedDict()
_session_locks: dict[str, threading.Lock] = {}
MAX_SESSIONS = 100
MAX_BODY_BYTES = 25 * 1024 * 1024  # reject oversized POST bodies (audio turns are well under this)

GREETING = "Thanks for calling Vera Hotel reservations. How can I help?"


def _load_env_files() -> None:
    load_env_files((PIPELINE_ROOT / ".env", ROOT / ".env"))


def _agent_provider_name() -> str:
    return os.getenv("PROVIDER", "mock").lower()


def _livekit_url() -> str:
    raw = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    if raw.startswith("http://"):
        return "ws://" + raw[len("http://"):]
    if raw.startswith("https://"):
        return "wss://" + raw[len("https://"):]
    return raw


def _livekit_api_key() -> str:
    return os.getenv("LIVEKIT_API_KEY", "devkey")


def _livekit_api_secret() -> str:
    return os.getenv("LIVEKIT_API_SECRET", "secret")


def _livekit_room() -> str:
    return os.getenv("LIVEKIT_ROOM", "vera-demo-room")


def _new_agent():
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from agent import Agent
    from providers import make_provider

    return Agent(make_provider(_agent_provider_name()))


def _get_session(session_id: str):
    with _session_registry_lock:
        if session_id not in _agent_sessions:
            _agent_sessions[session_id] = _new_agent()
            _session_locks[session_id] = threading.Lock()
        else:
            _agent_sessions.move_to_end(session_id)  # mark most-recently-used
        # Evict least-recently-used sessions beyond the cap, skipping any whose lock
        # is currently held (a turn is in flight) and the session we just returned.
        while len(_agent_sessions) > MAX_SESSIONS:
            for old_id in list(_agent_sessions):  # oldest first
                if old_id == session_id:
                    continue
                lock = _session_locks.get(old_id)
                if lock is not None and lock.locked():
                    continue
                _agent_sessions.pop(old_id, None)
                _session_locks.pop(old_id, None)
                break
            else:
                break  # nothing evictable this pass
        return _agent_sessions[session_id], _session_locks[session_id]


def _reset_session(session_id: str) -> None:
    # Resets only the conversation, NOT the booking inventory: bookings persist for
    # the server's lifetime so a later caller can't rebook a taken room ("next
    # customer can't book it"). Restart the server for a fresh booking demo.
    with _session_registry_lock:
        _agent_sessions.pop(session_id, None)
        _session_locks.pop(session_id, None)


def _trace(session_id: str, turn_id: str | None = None):
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from telemetry import TurnTrace

    return TurnTrace(session_id=session_id, turn_id=turn_id)


def _finish_response(agent, trace, reply: str, action: str | None, **extra) -> dict:
    from telemetry import write_trace

    sources = extra.pop("response_sources", agent.last_sources)
    payload = trace.finish(action=action, sources=sources)
    write_trace(payload)
    return {
        "reply": reply,
        "action": action,
        "provider": getattr(agent.provider, "name", _agent_provider_name()),
        "model": getattr(agent.provider, "llm_model", "unknown"),
        "language": agent.current_language,
        "locale": agent.current_locale,
        "sources": sources,
        "trace": payload,
        # Pipeline mode for the UI badge. The browser turn is the BATCH cascade
        # (one JSON response with the full reply + audio). Streaming (Option A,
        # sentence-by-sentence) currently lives in the CLI voice_loop; when the
        # browser transport is streamed (Phase 2) this becomes "streaming".
        "pipeline": "batch",
        **extra,
    }


def _browser_tts_payload(agent, trace, text: str) -> dict:
    """Return provider audio for the browser or select its local voice fallback."""
    provider = agent.provider
    backend = getattr(provider, "tts_backend", "provider")
    if backend != "provider" or getattr(provider, "name", "") == "mock":
        return {"ttsBackend": "browser"}

    language = getattr(agent, "current_language", None)  # per-turn native voice (Deepgram)
    model = getattr(provider, "tts_model", "unknown")
    # voice_for reports the ACTUAL per-turn voice (e.g. Deepgram picks aura-2-celeste-es
    # for Spanish); the model label stays the configured TTS model. When TTS runs on the
    # base provider (e.g. STT-only Deepgram) voice_for returns that base voice, so model
    # and voice each stay correct instead of collapsing onto the voice.
    resolve = getattr(provider, "voice_for", None)
    voice = resolve(language) if resolve else getattr(provider, "tts_voice", "unknown")
    try:
        with trace.span("tts", model=model, voice=voice):
            audio = provider.synthesize(text, language)
    except Exception as exc:
        trace.event("tts.fallback", errorType=type(exc).__name__)
        return {"ttsBackend": "browser", "ttsFallback": True}

    if not audio:
        trace.event("tts.fallback", errorType="EmptyAudio")
        return {"ttsBackend": "browser", "ttsFallback": True}
    return {
        "ttsBackend": "provider",
        "ttsModel": model,
        "ttsVoice": voice,
        "audioContentType": "audio/wav",
        "audioBase64": base64.b64encode(audio).decode("ascii"),
    }


def _greeting_reply(session_id: str) -> dict:
    agent, lock = _get_session(session_id)
    trace = _trace(session_id, "greeting")
    trace.event("greeting.requested")
    # _finish_response reads live agent state (language/locale/sources), so it
    # must run under the session lock -- a queued same-session turn could
    # otherwise mutate that state between our turn and its final payload.
    with lock:
        tts = _browser_tts_payload(agent, trace, GREETING)
        return _finish_response(
            agent,
            trace,
            GREETING,
            None,
            response_sources=[],
            **tts,
        )


def _agent_reply(text: str, session_id: str, turn_id: str | None) -> dict:
    agent, lock = _get_session(session_id)
    trace = _trace(session_id, turn_id)
    trace.event("input.text")
    with lock:
        reply, action = agent.respond(text, trace=trace)
        tts = _browser_tts_payload(agent, trace, reply)
        return _finish_response(agent, trace, reply, action, **tts)


def _transcribe_turn(agent, trace, audio: bytes, content_type: str,
                     was_barge_in: bool):
    """STT + suppression for one browser voice turn (call under the session lock).

    Returns (transcript, None) when the turn should proceed, or
    (transcript, ignored_final_payload) when it must be suppressed. Extracted
    verbatim from the batch path so batch and streaming share ONE source of
    truth for noise / hallucination / echo suppression.
    """
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from providers import is_noise_transcript, is_stt_hallucination

    # One uniform STT call across mock / OpenAI / Groq / Deepgram. The provider
    # owns container handling (transcribe_media), so no OpenAI-SDK reach-in here.
    with trace.span("stt", model=getattr(agent.provider, "stt_model", "unknown")):
        try:
            transcript = agent.provider.transcribe_media(audio, content_type)
        except Exception as exc:
            # A speech-vendor error (bad key, 429, timeout) degrades the turn
            # instead of 500-ing the browser and leaking the raw error string.
            trace.event("stt.error", errorType=type(exc).__name__)
            return "", _finish_response(
                agent, trace, "", None,
                transcript="",
                sttModel=getattr(agent.provider, "stt_model", "unknown"),
                ignored=True, ignoreReason="stt_error", response_sources=[],
            )
    if is_noise_transcript(transcript) or is_stt_hallucination(transcript):
        trace.event("stt.suppressed", transcript=transcript)
        return transcript, _finish_response(
            agent, trace, "", None,
            transcript=transcript,
            sttModel=getattr(agent.provider, "stt_model", "unknown"),
            ignored=True, ignoreReason="noise_or_hallucination", response_sources=[],
        )
    if (was_barge_in and _is_probable_playback_echo(transcript)
            and _normalize_echo(transcript) in _normalize_echo(_last_agent_reply(agent))):
        # Suppress only a courtesy phrase the agent ACTUALLY just spoke (a real echo);
        # a genuine "thanks" interruption the agent never said still reaches respond().
        trace.event("barge_in.echo_suppressed", transcript=transcript)
        return transcript, _finish_response(
            agent, trace, "", None,
            transcript=transcript,
            sttModel=getattr(agent.provider, "stt_model", "unknown"),
            ignored=True, ignoreReason="probable_playback_echo", response_sources=[],
        )
    return transcript, None


def _voice_agent_reply(
    audio: bytes,
    content_type: str,
    session_id: str,
    turn_id: str | None,
    was_barge_in: bool,
) -> dict:
    agent, lock = _get_session(session_id)
    trace = _trace(session_id, turn_id)
    trace.event("audio.received", bytes=len(audio), contentType=content_type)
    if was_barge_in:
        trace.event("barge_in.turn_started")
    with lock:
        transcript, ignored = _transcribe_turn(agent, trace, audio, content_type, was_barge_in)
        if ignored is not None:
            return ignored
        reply, action = agent.respond(transcript, trace=trace)
        tts = _browser_tts_payload(agent, trace, reply)
        # Inside the lock: _finish_response reads live agent state (language/
        # locale/sources) that the NEXT queued turn's respond() would reset.
        return _finish_response(
            agent,
            trace,
            reply,
            action,
            transcript=transcript,
            sttModel=getattr(agent.provider, "stt_model", "unknown"),
            **tts,
        )


def _browser_streaming_enabled(agent) -> bool:
    """True when this turn should stream sentence-chunked audio to the browser.
    Reuses the CLI's gate verbatim: STREAMING env (default off) AND the provider
    supports chat_stream AND cloud TTS -- so mock / TTS_BACKEND=system / default
    config all stay on the batch path."""
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from streaming import streaming_enabled

    return streaming_enabled(agent.provider)


def _stream_voice_agent_reply(
    audio: bytes,
    content_type: str,
    session_id: str,
    turn_id: str | None,
    was_barge_in: bool,
    emit,
) -> dict:
    """Streaming twin of _voice_agent_reply: NDJSON events via `emit(dict) -> bool`.

    Event order: transcript -> (sentence | reset)* -> final. Mirrors the CLI's
    _respond_streaming semantics (turn-scoped first_audio, on_reset drops the
    unspoken segment, the degrade reply is always sent). `emit` returns False once
    the client is gone; from then on TTS is skipped (no wasted spend) but
    agent.respond runs to completion so history/trace stay coherent.
    """
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from streaming import SentenceBuffer

    agent, lock = _get_session(session_id)
    trace = _trace(session_id, turn_id)
    trace.event("audio.received", bytes=len(audio), contentType=content_type)
    if was_barge_in:
        trace.event("barge_in.turn_started")
    with lock:
        transcript, ignored = _transcribe_turn(agent, trace, audio, content_type, was_barge_in)
        if ignored is not None:
            emit({"type": "final", **ignored})
            return ignored

        buffer = SentenceBuffer()
        state = {"spoke": False, "first_audio": False, "seq": 0, "segment": 0,
                 "audio_sent": False, "client_gone": False}
        if not emit({
            "type": "transcript",
            "transcript": transcript,
            "sttModel": getattr(agent.provider, "stt_model", "unknown"),
        }):
            state["client_gone"] = True  # client already left: skip all TTS spend

        def emit_sentence(sentence: str) -> None:
            # MUST NEVER RAISE: an exception inside on_delta propagates into
            # agent.respond's provider-error handler and falsely degrades the
            # whole turn to a transfer (agent.py). TTS failure -> empty audio.
            audio_b64 = ""
            if not state["client_gone"]:
                try:
                    with trace.span("tts", model=getattr(agent.provider, "tts_model", "unknown")):
                        wav = agent.provider.synthesize(sentence, agent.current_language)
                    if wav:
                        audio_b64 = base64.b64encode(wav).decode("ascii")
                except Exception as exc:
                    trace.event("tts.sentence_fallback", errorType=type(exc).__name__)
            if audio_b64 and not state["first_audio"]:
                trace.event("tts.first_audio", model=getattr(agent.provider, "tts_model", "unknown"))
                state["first_audio"] = True
            state["spoke"] = True
            state["audio_sent"] = state["audio_sent"] or bool(audio_b64)
            ok = emit({
                "type": "sentence",
                "seq": state["seq"],
                "segment": state["segment"],
                "text": sentence,
                "audioBase64": audio_b64,
                "audioContentType": "audio/wav",
            })
            state["seq"] += 1
            if not ok:
                state["client_gone"] = True

        def on_delta(delta: str) -> None:
            for sentence in buffer.feed(delta):
                emit_sentence(sentence)

        def on_reset() -> None:
            buffer.flush()  # discard the unspoken part of the abandoned segment
            state["spoke"] = False
            state["segment"] += 1
            if not state["client_gone"]:
                if not emit({"type": "reset", "segment": state["segment"]}):
                    state["client_gone"] = True

        try:
            reply, action = agent.respond(transcript, trace=trace,
                                          on_delta=on_delta, on_reset=on_reset)
            tail = buffer.flush()
            if tail:
                emit_sentence(tail)
            if not state["spoke"] and reply.strip():
                # Final reply never streamed (degrade replaced it, or a bare
                # non-sentence) -- send it whole so the caller always hears it.
                emit_sentence(reply)
        except Exception as exc:
            # Unexpected mid-stream server failure: persist the trace and give
            # the client a well-formed final instead of a truncated stream.
            trace.event("stream.error", errorType=type(exc).__name__)
            final = _finish_response(
                agent, trace, "", None,
                transcript=transcript,
                sttModel=getattr(agent.provider, "stt_model", "unknown"),
                ignored=True, ignoreReason="stream_error", response_sources=[],
            )
            emit({"type": "final", **final})
            return final

        if state["audio_sent"]:
            tts_meta = {
                "ttsBackend": "provider",
                "ttsModel": getattr(agent.provider, "tts_model", "unknown"),
                "ttsVoice": (agent.provider.voice_for(agent.current_language)
                             if getattr(agent.provider, "voice_for", None)
                             else getattr(agent.provider, "tts_voice", "unknown")),
            }
        else:
            tts_meta = {"ttsBackend": "browser", "ttsFallback": True}

        # Inside the lock: _finish_response reads live agent state (language/
        # locale/sources) that the NEXT queued turn's respond() would reset.
        final = _finish_response(
            agent, trace, reply, action,
            transcript=transcript,
            sttModel=getattr(agent.provider, "stt_model", "unknown"),
            # honest badge: "streaming" only if at least one audio chunk streamed
            pipeline="streaming" if state["audio_sent"] else "batch",
            sentences=state["seq"],
            **tts_meta,
        )
        emit({"type": "final", **final})
        return final


def _normalize_echo(text: str) -> str:
    """Lowercase + strip ALL punctuation (incl. ! ? that smart_format restores) so a
    spoken-echo transcript matches the reply text it came from."""
    stripped = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(stripped.split())


def _is_probable_playback_echo(transcript: str) -> bool:
    return _normalize_echo(transcript) in {
        "all right",
        "alright",
        "thanks",
        "thank you",
        "youre welcome",
        "your welcome",
    }


def _last_agent_reply(agent) -> str:
    """The text the agent most recently spoke (what a playback echo would repeat)."""
    for message in reversed(getattr(agent, "messages", None) or []):
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def _token(identity: str, name: str, room: str) -> str:
    if _livekit_api_secret() == "secret":
        warnings.filterwarnings("ignore", category=jwt.InsecureKeyLengthWarning)
    return (
        api.AccessToken(_livekit_api_key(), _livekit_api_secret())
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/web/index.html"
            return super().do_GET()
        if parsed.path == "/state":
            return self._send_json({
                "livekitRoom": _livekit_room(),
                "livekitUrl": _livekit_url(),
                "agentProvider": _agent_provider_name(),
                "languages": ["en", "es"],
            })
        if parsed.path != "/token":
            # Only static assets under /web/ are served. The handler's directory is the
            # livekit/ root, so an unrestricted GET would expose /.env and /talk_server.py
            # (SimpleHTTPRequestHandler filters '..' but NOT dotfiles) to any client.
            if parsed.path.startswith("/web/"):
                return super().do_GET()
            return self.send_error(404)

        query = parse_qs(parsed.query)
        identity = query.get("identity", ["caller-demo"])[0]
        name = query.get("name", [identity])[0]
        room = query.get("room", [_livekit_room()])[0]

        payload = {
            "url": _livekit_url(),
            "room": room,
            "identity": identity,
            "token": _token(identity, name, room),
        }
        self._send_json(payload)

    def _read_length(self) -> "int | None":
        """Parse + bound Content-Length; send an error and return None if invalid."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=400)
            return None
        if length < 0 or length > MAX_BODY_BYTES:
            self._send_json({"error": "Request body too large"}, status=413)
            return None
        return length

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        session_id = self.headers.get("X-Session-ID", "browser-demo")
        turn_id = self.headers.get("X-Turn-ID")
        if parsed.path == "/reset":
            _reset_session(session_id)
            return self._send_json({"reset": True, "sessionId": session_id})
        if parsed.path == "/greeting":
            try:
                return self._send_json(_greeting_reply(session_id))
            except Exception as exc:
                return self._send_json({"error": str(exc)}, status=500)
        if parsed.path == "/voice-agent":
            return self._handle_voice_agent(session_id, turn_id)
        if parsed.path != "/agent":
            self.send_error(404, "File not found")
            return

        try:
            length = self._read_length()
            if length is None:
                return
            body = self.rfile.read(length)
            payload = json.loads(body or b"{}")
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("Missing text")
            response = _agent_reply(text, session_id, turn_id)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json(response)

    def _handle_voice_agent(self, session_id: str, turn_id: str | None) -> None:
        try:
            length = self._read_length()
            if length is None:
                return
            audio = self.rfile.read(length)
            if not audio:
                raise ValueError("Missing audio")
            content_type = self.headers.get("Content-Type", "")
            barge = self.headers.get("X-Barge-In", "false").lower() == "true"
            agent, _lock = _get_session(session_id)
            if _browser_streaming_enabled(agent):
                return self._stream_voice_agent(audio, content_type, session_id, turn_id, barge)
            response = _voice_agent_reply(audio, content_type, session_id, turn_id, barge)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json(response)

    def _stream_voice_agent(self, audio, content_type, session_id, turn_id, barge) -> None:
        # HTTP/1.0 + no Content-Length = close-delimited body: each event line is
        # flushed as it happens and the socket close marks end-of-stream. The
        # client detects streaming mode purely from this Content-Type.
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return  # client vanished before headers; nothing to stream
        try:
            _stream_voice_agent_reply(
                audio, content_type, session_id, turn_id, barge,
                emit=self._write_stream_event,
            )
        except Exception as exc:
            # Headers are already sent -- degrade in-band; the missing `final`
            # tells the client the stream was truncated.
            self._write_stream_event({"type": "error", "error": type(exc).__name__})

    def _write_stream_event(self, event: dict) -> bool:
        """Write one NDJSON event; False once the client is gone (barge-in abort,
        tab close) so the producer can stop spending on TTS."""
        try:
            self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    _load_env_files()
    os.environ.setdefault(
        "TELEMETRY_JSONL",
        str(REPO_ROOT / "logs" / "voice-events.jsonl"),
    )
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}")
    print(f"LiveKit URL: {_livekit_url()}")
    print(f"Room: {_livekit_room()}")
    print(f"Agent provider: {_agent_provider_name()}")
    print(f"TTS backend: {os.getenv('TTS_BACKEND', 'provider').lower()}")
    print("Use the two panes for LiveKit audio. Use the conversation panel for the hotel agent.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
