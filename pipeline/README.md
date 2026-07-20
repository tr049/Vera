# Vera Voice Pipeline

The pipeline is the local text and microphone runtime:

```text
microphone -> WebRTC VAD -> STT -> AgentRouter -> LLM -> RAG and tools -> TTS
```

## Setup

```bash
# Dependencies are managed by ../pyproject.toml + ../uv.lock via uv (works from any repo dir).
uv sync                              # offline/mock only — installs nothing (empty venv by design)
uv sync --extra live                 # + Groq/OpenAI provider (openai, python-dotenv)
uv sync --extra audio --extra live   # + real microphone (sounddevice, webrtcvad, numpy)
cp config.example.env .env
```

Select `PROVIDER=mock`, `PROVIDER=openai`, or `PROVIDER=groq` in `.env` (this picks the LLM). Only a live provider requires an API key. Optionally set `STT_PROVIDER=deepgram` and/or `TTS_PROVIDER=deepgram` to route just those stages to Deepgram (Nova-3 / Aura-2) — needs `DEEPGRAM_API_KEY` and `uv sync --extra live --extra deepgram`.

## Verify Offline

```bash
uv run python smoke_test.py
uv run python -m unittest -v test_features.py
PROVIDER=mock uv run python voice_loop.py --text
```

## Run Live

```bash
uv run --extra live python voice_loop.py --text
uv run --extra audio --extra live python voice_loop.py
```

Text mode verifies the model, tools, RAG, routing, and guardrails without microphone uncertainty. Voice mode adds local audio capture, endpointing, STT, TTS, and stage telemetry.

## Supporting Commands

Evaluate deterministic behavior:

```bash
cd ../evals
uv run python run_evals.py --suite all
```

Estimate capacity:

```bash
uv run python scale_check.py --dau 1000000
```

## Modules

| File | Responsibility |
|------|----------------|
| `agent.py` | Prompt, tools, RAG tool, routing integration, and conversation state |
| `providers.py` | Mock, OpenAI, and Groq adapters |
| `router.py` | English and Spanish session routing |
| `knowledge.py` | Local SQLite FTS5 retrieval with bilingual query expansion |
| `telemetry.py` | Structured turn events, timings, and optional JSONL output |
| `voice_loop.py` | Text or microphone turn loop and TTS playback |
| `scale_check.py` | DAU, concurrency, worker, request-rate, and cost model |
| `smoke_test.py` | Basic offline end-to-end assertion |
| `test_features.py` | Focused router, RAG, telemetry, and scale tests |

Use `TTS_BACKEND=system` during development to avoid cloud TTS cost. Verify current provider pricing and limits before estimating development or production spend.
