"""
voice_loop.py  -  the turn loop (Layer A).

    mic -> VAD endpointing -> STT -> Agent -> TTS -> speakers

with per-stage latency timing so the room can SEE where the ~800ms turn budget
goes. Provider (Groq/OpenAI) is chosen in .env; see providers.py.

Modes:
    python voice_loop.py          # real mic
    python voice_loop.py --text   # type your turn (no audio deps / no mic)  -  always works
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import uuid

from agent import Agent
from inventory import reset_inventory
from providers import is_noise_transcript, is_stt_hallucination, make_provider
from streaming import SentenceBuffer, streaming_enabled
from telemetry import TurnTrace, format_trace, write_trace

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass  # .env is optional; env vars still work. Keeps the offline mock zero-install.

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
ENDPOINT_SILENCE_MS = int(os.getenv("ENDPOINT_SILENCE_MS", "600"))
# Minimum voiced audio for a real turn. Below this the capture is background
# noise, not speech -> drop it so STT never hallucinates gibberish from it.
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "200"))


# --- Audio (imported lazily so --text mode needs no audio libs) ---

def record_utterance(trace: TurnTrace) -> bytes:
    """Capture mic until the caller pauses (VAD endpointing). Returns 16-bit PCM."""
    import sounddevice as sd
    import webrtcvad

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_ms = 30
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)     # samples per frame
    silence_frames_needed = ENDPOINT_SILENCE_MS // frame_ms
    min_speech_frames = max(1, MIN_SPEECH_MS // frame_ms)

    frames: list[bytes] = []
    started = False
    trailing_silence = 0
    speech_frames = 0

    print("  (listening: speak, then pause)")
    trace.event("vad.listening", aggressiveness=VAD_AGGRESSIVENESS)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=frame_len,
                           dtype="int16", channels=1) as stream:
        while True:
            block, _ = stream.read(frame_len)
            frame = bytes(block)
            if len(frame) < frame_len * 2:             # short tail frame
                continue
            speech = vad.is_speech(frame, SAMPLE_RATE)
            if speech:
                if not started:
                    trace.event("vad.speech_started")
                started = True
                trailing_silence = 0
                speech_frames += 1
                frames.append(frame)
            elif started:
                trailing_silence += 1
                frames.append(frame)
                if trailing_silence >= silence_frames_needed:
                    trace.event(
                        "vad.endpoint_detected",
                        endpointSilenceMs=ENDPOINT_SILENCE_MS,
                    )
                    break
    if speech_frames < min_speech_frames:
        # Too little real speech -> treat as background noise, not a turn.
        trace.event("vad.discarded_noise", speechFrames=speech_frames)
        return b""
    return b"".join(frames)


def play_wav_bytes(wav: bytes) -> None:
    """Play WAV bytes via the configured local audio player."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(wav)
        f.flush()
        subprocess.run([os.getenv("AUDIO_PLAYER_CMD", "afplay"), f.name], check=False)


def speak(provider: Provider, text: str, language: str | None = None) -> None:
    """Speak `text`: cloud TTS returns audio, or the provider handles playback.
    `language` lets a per-language TTS (Deepgram) pick the native voice."""
    print(f"agent> {text}")
    audio = provider.synthesize(text, language)
    if audio:
        play_wav_bytes(audio)


def _play_sentence(provider: Provider, sentence: str, language: str | None) -> None:
    """Speak one streamed sentence. Exception-safe by design: this runs inside
    the LLM stream's on_delta callback, so a TTS error must NOT propagate (it
    would wrongly degrade the turn to a transfer). Swallow it with a warning,
    mirroring the batch path's TTS guard."""
    print(f"agent> {sentence}")
    try:
        audio = provider.synthesize(sentence, language)
        if audio:
            play_wav_bytes(audio)
    except Exception as exc:
        print(f"  [TTS playback failed: {type(exc).__name__}] {exc}")


def _respond_streaming(provider: Provider, agent: Agent, user_text: str,
                       trace: TurnTrace) -> tuple[str, str | None]:
    """Streaming turn: play each sentence as the model produces it, cutting
    time-to-first-audio. Returns (reply, action) exactly like the batch path,
    with the full reply text unchanged.

    The buffer + `spoke_reply` flag track only the CURRENT reply segment.
    `on_reset` (fired by respond at a tool call or a degrade) discards a buffered
    preamble/partial so it can neither fuse with the final reply nor suppress a
    degrade apology. TTS runs inside the exception-safe _play_sentence, so a
    playback error never degrades the turn."""
    buffer = SentenceBuffer()
    # spoke_reply tracks the CURRENT segment (on_reset clears it); first_audio is
    # turn-scoped so the latency event fires once even across a preamble + reply.
    state = {"spoke_reply": False, "first_audio": False}

    def emit(sentence: str) -> None:
        if not state["first_audio"]:
            trace.event("tts.first_audio", model=getattr(provider, "tts_model", "unknown"))
            state["first_audio"] = True
        state["spoke_reply"] = True
        _play_sentence(provider, sentence, agent.current_language)

    def on_delta(delta: str) -> None:
        for sentence in buffer.feed(delta):
            emit(sentence)

    def on_reset() -> None:
        # This segment (a preamble, or an abandoned partial before a degrade) is
        # not part of the final reply -- drop it and start the next one clean.
        buffer.flush()
        state["spoke_reply"] = False

    reply, action = agent.respond(user_text, trace=trace, on_delta=on_delta, on_reset=on_reset)
    tail = buffer.flush()
    if tail:
        emit(tail)
    if not state["spoke_reply"] and reply.strip():
        # Final reply never streamed (degrade replaced it, or a bare non-sentence)
        # -- speak it whole so the caller always hears the returned reply.
        emit(reply)
    return reply, action


# --- The loop ---

def run(text_mode: bool) -> None:
    provider = make_provider()
    reset_inventory()  # fresh booking store for this call
    agent = Agent(provider)
    stream = streaming_enabled(provider, text_mode=text_mode)  # opt-in, off by default
    session_id = f"cli-{uuid.uuid4().hex[:10]}"
    print(f"Provider: {provider.name} | LLM: {provider.llm_model}"
          f"{' | streaming' if stream else ''}")
    print("Call started. Say/type 'goodbye' or Ctrl-C to hang up.\n")

    try:
        speak(provider, "Thanks for calling Vera Hotel reservations. How can I help?")
    except Exception as exc:  # a TTS-vendor/network error must not abort startup
        print(f"  [greeting playback failed: {type(exc).__name__}] {exc}")

    while True:
        try:
            if text_mode:
                user_text = input("you> ")
                trace = TurnTrace(session_id=session_id)
                trace.event("input.text")
            else:
                trace = TurnTrace(session_id=session_id)
                with trace.span("capture"):
                    pcm = record_utterance(trace)
                if not pcm:
                    continue  # capture was background noise; keep listening
                with trace.span("stt", model=getattr(provider, "stt_model", "unknown")):
                    user_text = provider.transcribe(pcm, SAMPLE_RATE)
                print(f"you> {user_text}")
                if is_noise_transcript(user_text) or is_stt_hallucination(user_text):
                    trace.event("stt.suppressed", transcript=user_text)
                    continue  # STT noise/echo hallucination; skip the turn
            if not user_text.strip():
                continue

            if stream:
                # Streaming path: sentences are spoken as the model produces them.
                reply, action = _respond_streaming(provider, agent, user_text, trace)
            else:
                reply, action = agent.respond(user_text, trace=trace)
                # Wrap TTS on its own so a playback error can't drop a pending
                # hangup/transfer (the call must still end).
                try:
                    with trace.span("tts", model=getattr(provider, "tts_model", "unknown")):
                        speak(provider, reply, agent.current_language)
                except Exception as exc:
                    print(f"  [TTS playback failed: {type(exc).__name__}] {exc}")

            payload = trace.finish(action=action, sources=agent.last_sources)
            write_trace(payload)
            print(format_trace(payload))
            print()

            if action == "hangup":
                print("[call ended: SIP BYE]")
                break
            if action == "transfer":
                print("[transferring to front desk: SIP REFER to front-desk]")
                break

        except (EOFError, KeyboardInterrupt):
            print("\n[caller hung up: SIP BYE]")
            break
        except ImportError:
            raise  # missing audio deps (e.g. no --extra audio) are fatal, not a retryable turn
        except Exception as exc:
            # A speech-vendor / network error (bad key, 429, timeout) degrades the
            # turn instead of crashing the whole call.
            print(f"  [turn error: {type(exc).__name__}] {exc} — please try again.\n")
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Vera voice loop")
    parser.add_argument("--text", action="store_true",
                        help="type turns instead of speaking (no mic / no audio deps)")
    args = parser.parse_args()
    run(text_mode=args.text)


if __name__ == "__main__":
    main()
