import { Room, RoomEvent, Track } from "/node_modules/livekit-client/dist/livekit-client.esm.mjs";

// Motion (motion.dev) is OPTIONAL polish, loaded lazily below so a missing or
// failed vendor file can NEVER brick the app -- the UI just runs without the
// spring choreography. `animate`/`stagger` stay null until (and unless) it loads.
let animate = null;
let stagger = null;

const callerRoot = document.querySelector('[data-client="caller"]');
const agentRoot = document.querySelector('[data-client="agent"]');
const callerStatus = callerRoot.querySelector('[data-role="status"]');
const agentStatus = agentRoot.querySelector('[data-role="status"]');
const startButton = document.querySelector("#start-call");
const muteButton = document.querySelector("#mute-call");
const endButton = document.querySelector("#end-call");
const participantsEl = document.querySelector("#participants");
const providerEl = document.querySelector("#provider");
const languageEl = document.querySelector("#language");
const transcriptEl = document.querySelector("#transcript");
const sourcesEl = document.querySelector("#sources");
const voiceStatusEl = document.querySelector("#voice-status");
const listeningStateEl = document.querySelector("#listening-state");
const eventsEl = document.querySelector("#events");
const pipelineEl = document.querySelector("#pipeline");
const endpointControl = document.querySelector("#endpoint-control");
const endpointValue = document.querySelector("#endpoint-value");
const sensitivityControl = document.querySelector("#sensitivity-control");
const sensitivityValue = document.querySelector("#sensitivity-value");
const vadReadout = document.querySelector("#vad-readout");

const metrics = {
  stt: document.querySelector("#metric-stt"),
  llm: document.querySelector("#metric-llm"),
  tools: document.querySelector("#metric-tools"),
  total: document.querySelector("#metric-total"),
  firstAudio: document.querySelector("#metric-first-audio"),
  barge: document.querySelector("#metric-barge"),
};

const voiceOrb = document.querySelector(".voice-orb");

// Flash a metric tile when its value changes -- self-contained, so the many
// `metrics.*.textContent = ...` assignment sites need no edits. Reduced-motion
// keeps the colour flash (CSS) and drops the translate.
for (const metricEl of Object.values(metrics)) {
  const tile = metricEl?.closest("div");
  if (!tile) continue;
  let last = metricEl.textContent;
  new MutationObserver(() => {
    if (metricEl.textContent === last) return; // don't flash on identical rewrites
    last = metricEl.textContent;
    tile.classList.remove("bump");
    void tile.offsetWidth; // reflow so the animation restarts on every change
    tile.classList.add("bump");
  }).observe(metricEl, { childList: true, characterData: true, subtree: true });
}

// -- Motion (motion.dev, vendored + offline) for one-shot spring choreography.
// The continuous voice-state loops stay pure CSS (compositor); Motion is used
// only for enters/reveals so it never competes with the realtime audio loop.
const pipelineModeEl = document.querySelector("#pipeline-mode");
const pipelineModeLabel = document.querySelector("#pipeline-mode-label");
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Lazy-load Motion, then run the entrance. Transform-only (panels keep opacity 1)
// so a throttled/backgrounded tab that stalls WAAPI can never leave the UI hidden;
// a short fail-safe cancels any lingering animation so nothing sticks at the offset.
if (!REDUCED_MOTION) {
  import("/web/vendor/motion.min.mjs")
    .then((motion) => {
      animate = motion.animate;
      stagger = motion.stagger;
      animate("main > *", { y: [18, 0] },
        { delay: stagger(0.05), duration: 0.5, ease: [0.05, 0.7, 0.1, 1] });
      window.setTimeout(() => {
        document.querySelectorAll("main > *").forEach((el) =>
          el.getAnimations().forEach((a) => a.cancel()));
      }, 1200);
    })
    .catch(() => {}); // Motion unavailable -> no spring polish; the app still works
}

// Spring a newly-inserted transcript bubble in (side-aware). Reduced-motion:
// skipped, so the bubble just appears (its default opacity is 1).
function springInBubble(node, role) {
  if (REDUCED_MOTION || !animate) return;  // no Motion loaded -> bubble just appears
  const fromX = role === "caller" ? -14 : role === "agent" ? 14 : 0;
  // transform-only (no opacity) so the bubble is always visible even if throttled.
  animate(
    node,
    { x: [fromX, 0], y: [12, 0], scale: [0.96, 1] },
    { type: "spring", stiffness: 300, damping: 24 },
  );
}

// Reflect the response pipeline (batch vs streaming) in the header badge.
function setPipelineMode(mode) {
  if (!pipelineModeEl) return;
  const streaming = mode === "streaming";
  pipelineModeEl.dataset.mode = streaming ? "streaming" : "batch";
  if (pipelineModeLabel) {
    pipelineModeLabel.textContent = streaming ? "Streaming cascade" : "Batch cascade";
  }
}

const sessionId = `browser-${crypto.randomUUID()}`;
let turnCounter = 0;
let callerRoom = null;
let agentRoom = null;
let listenStream = null;
let audioContext = null;
let analyser = null;
let vadFrame = null;
let recorder = null;
let recordedChunks = [];
let recordingStartedAt = 0;
let lastSpeechAt = 0;
let speechCandidateAt = 0;
let bargeCandidateAt = 0;
let listenCooldownUntil = 0;
let agentBusy = false;
let agentSpeaking = false;
let muted = false;
let noiseFloor = 0.008;
let smoothedLevel = 0;
let discardRecording = false;
let currentTrace = null;
let lastEndpointAt = 0;
let playbackStartedAt = 0;
let playbackEchoFloor = 0.012;
let pendingBargeInTurn = false;
let currentTurnWasBargeIn = false;
let activeAgentAudio = null;
let playbackToken = 0;
let bargeRecordingCandidate = false;

const tuning = {
  endpointSilenceMs: 650,
  sensitivity: 3.2,
  minTurnMs: 500,
  // Confirmation windows balance "noise shouldn't trigger it" vs "my voice should".
  // speechConfirmationMs gates a NEW turn; bargeInConfirmationMs gates interrupting
  // the agent mid-sentence — kept short so a spoken word actually breaks through
  // (over a speaker the echo raises the bar; headphones make barge-in reliable).
  speechConfirmationMs: 200,
  bargeInConfirmationMs: 220,
  bargeInArmMs: 400,
  maxTurnMs: 20000,
};

function setCallControls(connected) {
  startButton.disabled = connected;
  muteButton.disabled = !connected;
  endButton.disabled = !connected;
  callerRoot.classList.toggle("connected", connected);
  agentRoot.classList.toggle("connected", connected);
}

// Every voice-state transition already flows through setListeningState, so this
// one chokepoint drives the hero orb's motion-shape (idle/listening/thinking/
// speaking/bargein/muted/error) with no changes to the turn/VAD logic.
const VOICE_ORB_STATE = {
  Idle: "idle",
  Calibrating: "idle",
  Listening: "listening",
  "Caller speaking": "listening",
  Processing: "thinking",
  "Agent speaking": "speaking",
  Interrupted: "bargein",
  Muted: "muted",
  Error: "error",
  "Connection failed": "error",
  "Mute failed": "error",
};

function setListeningState(state, detail) {
  listeningStateEl.textContent = state;
  voiceStatusEl.textContent = detail;
  if (voiceOrb) voiceOrb.dataset.voiceState = VOICE_ORB_STATE[state] || "idle";
}

function addTranscript(role, text, meta = "") {
  transcriptEl.querySelector(".empty")?.remove();
  const item = document.createElement("div");
  item.className = `bubble ${role}`;
  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = `${role === "caller" ? "Caller Demo" : "Vera Agent"}${meta ? ` | ${meta}` : ""}`;
  const body = document.createElement("div");
  body.textContent = text;
  item.append(label, body);
  transcriptEl.appendChild(item);
  springInBubble(item, role);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return item;
}

function addInterruption() {
  transcriptEl.querySelector(".empty")?.remove();
  const item = document.createElement("div");
  item.className = "bubble interruption";
  item.textContent = "Caller interrupted agent playback";
  transcriptEl.appendChild(item);
  springInBubble(item, "interruption");
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function formatMs(value) {
  return `${Math.round(value || 0)} ms`;
}

function eventDetail(event) {
  const attributes = event.attributes || {};
  if (attributes.tool) return `${event.name} | ${attributes.tool}`;
  if (attributes.language) return `${event.name} | ${attributes.language}`;
  if (attributes.durationMs !== undefined) return `${event.name} | ${formatMs(attributes.durationMs)}`;
  return event.name;
}

function renderTrace(trace) {
  currentTrace = trace;
  const timings = trace.timings || {};
  metrics.stt.textContent = formatMs(timings.stt);
  metrics.llm.textContent = formatMs(timings.llm);
  metrics.tools.textContent = formatMs(timings.tools);
  metrics.total.textContent = formatMs(trace.totalMs);

  for (const element of pipelineEl.querySelectorAll("[data-stage]")) {
    const stage = element.dataset.stage;
    const completed = stage === "vad" || timings[stage] !== undefined;
    element.classList.toggle("complete", completed);
  }

  eventsEl.innerHTML = "";
  for (const event of (trace.events || []).slice(-14)) {
    const row = document.createElement("div");
    row.className = "event-row";
    const time = document.createElement("time");
    time.textContent = `+${Math.round(event.offsetMs)}ms`;
    const detail = document.createElement("span");
    detail.textContent = eventDetail(event);
    row.append(time, detail);
    eventsEl.appendChild(row);
  }
}

function appendRuntimeEvent(name) {
  eventsEl.querySelector(".empty")?.remove();
  const row = document.createElement("div");
  row.className = "event-row";
  const time = document.createElement("time");
  time.textContent = "client";
  const detail = document.createElement("span");
  detail.textContent = name;
  row.append(time, detail);
  eventsEl.appendChild(row);
  eventsEl.scrollTop = eventsEl.scrollHeight;
}

function renderSources(sources) {
  sourcesEl.textContent = sources?.length
    ? sources.join(" | ")
    : "No retrieval used in the latest turn.";
}

function chooseVoice(locale) {
  if (!("speechSynthesis" in window)) return null;
  const language = locale.toLowerCase().split("-")[0];
  return window.speechSynthesis.getVoices().find(
    (voice) => voice.lang.toLowerCase().startsWith(language),
  ) || null;
}

function stopAgentPlayback() {
  playbackToken += 1;
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  if (activeAgentAudio) {
    activeAgentAudio.onplay = null;
    activeAgentAudio.onended = null;
    activeAgentAudio.onerror = null;
    activeAgentAudio.pause();
    activeAgentAudio.removeAttribute("src");
    activeAgentAudio = null;
  }
}

function beginAgentPlayback(token, backend) {
  if (token !== playbackToken) return;
  agentSpeaking = true;
  agentRoot.classList.add("speaking");
  playbackStartedAt = Date.now();
  playbackEchoFloor = Math.max(noiseFloor, 0.012);
  listenCooldownUntil = playbackStartedAt + tuning.bargeInArmMs;
  pipelineEl.querySelector('[data-stage="tts"]')?.classList.add("complete");
  appendRuntimeEvent(`tts.playback_started | ${backend}`);
  if (lastEndpointAt) {
    const firstAudioMs = Date.now() - lastEndpointAt;
    metrics.firstAudio.textContent = formatMs(firstAudioMs);
    appendRuntimeEvent(`turn.first_audio | ${formatMs(firstAudioMs)}`);
    lastEndpointAt = 0;
  }
  setListeningState("Agent speaking", "Interrupt naturally by speaking over Vera.");
}

function finishAgentPlayback(token) {
  if (token !== playbackToken) return;
  activeAgentAudio = null;
  agentSpeaking = false;
  agentRoot.classList.remove("speaking");
  listenCooldownUntil = Date.now() + 500;
  if (listenStream) {
    setListeningState("Listening", "Speak naturally. Vera can be interrupted while talking.");
  }
}

function speakWithBrowserVoice(text, locale, token) {
  if (!("speechSynthesis" in window) || token !== playbackToken) {
    finishAgentPlayback(token);
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = locale;
  utterance.rate = 0.98;
  utterance.pitch = 1.0;
  const voice = chooseVoice(locale);
  if (voice) utterance.voice = voice;

  utterance.onstart = () => beginAgentPlayback(token, "browser");
  utterance.onend = () => finishAgentPlayback(token);
  utterance.onerror = () => finishAgentPlayback(token);
  window.speechSynthesis.speak(utterance);
}

function speak(text, locale = "en-US", audioBase64 = "", audioContentType = "audio/wav") {
  stopAgentPlayback();
  const token = playbackToken;
  if (!audioBase64) {
    speakWithBrowserVoice(text, locale, token);
    return;
  }

  const audio = new Audio(`data:${audioContentType};base64,${audioBase64}`);
  activeAgentAudio = audio;
  let fellBack = false;
  const fallback = () => {
    if (fellBack || token !== playbackToken) return;
    fellBack = true;
    activeAgentAudio = null;
    appendRuntimeEvent("tts.provider_playback_failed | browser fallback");
    speakWithBrowserVoice(text, locale, token);
  };
  audio.onplay = () => beginAgentPlayback(token, "provider");
  audio.onended = () => finishAgentPlayback(token);
  audio.onerror = fallback;
  audio.play().catch(fallback);
}

function interruptAgent(detectedAt, turnAlreadyRecording = false) {
  if (!agentSpeaking) return;
  stopAgentPlayback();
  agentSpeaking = false;
  agentRoot.classList.remove("speaking");
  listenCooldownUntil = Date.now() + 80;
  pendingBargeInTurn = !turnAlreadyRecording;
  addInterruption();
  appendRuntimeEvent("barge_in.detected");
  metrics.barge.textContent = formatMs(Date.now() - detectedAt);
  setListeningState("Interrupted", "Vera stopped. Listening to the caller.");
}

function audioLevel() {
  if (!analyser) return 0;
  const samples = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(samples);
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return Math.sqrt(sum / samples.length);
}

function thresholds() {
  const start = Math.min(0.09, Math.max(0.012, noiseFloor * tuning.sensitivity));
  return {
    start,
    end: Math.max(0.008, start * 0.58),
    barge: Math.min(0.12, Math.max(0.024, start * 1.55, playbackEchoFloor * 1.8)),
  };
}

function startTurnRecording(isBargeIn = false) {
  if (!listenStream || recorder || agentBusy || muted) return;
  recordedChunks = [];
  discardRecording = false;
  recorder = new MediaRecorder(listenStream);
  currentTurnWasBargeIn = isBargeIn || pendingBargeInTurn;
  pendingBargeInTurn = false;
  recordingStartedAt = Date.now();
  lastSpeechAt = recordingStartedAt;
  callerRoot.classList.add("speaking");
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  };
  recorder.onstop = () => {
    const shouldDiscard = discardRecording;
    const mimeType = recorder.mimeType || "audio/webm";
    const audioBlob = new Blob(recordedChunks, { type: mimeType });
    recorder = null;
    recordedChunks = [];
    callerRoot.classList.remove("speaking");
    if (shouldDiscard || audioBlob.size < 800) {
      currentTurnWasBargeIn = false;
      if (agentSpeaking) {
        setListeningState("Agent speaking", "Interrupt naturally by speaking over Vera.");
      } else if (listenStream) {
        setListeningState("Listening", "Speak naturally. Vera can be interrupted while talking.");
      }
      return;
    }
    sendAudioToAgent(audioBlob);
  };
  recorder.start(100);
  setListeningState("Caller speaking", "Listening for the end of the turn.");
}

function stopTurnRecording(discard = false) {
  if (!recorder || recorder.state === "inactive") return;
  discardRecording = discard;
  recorder.stop();
}

async function sendAudioToAgent(audioBlob) {
  agentBusy = true;
  setListeningState("Processing", "Transcribing and running the hotel agent.");
  const voicePlaceholder = addTranscript("caller", "Voice turn", "transcribing");
  const pending = addTranscript("agent", "Processing turn", "STT -> Router -> RAG -> LLM -> Tools");
  const turnId = `turn-${++turnCounter}`;
  const wasBargeIn = currentTurnWasBargeIn;
  currentTurnWasBargeIn = false;

  try {
    const response = await fetch("/voice-agent", {
      method: "POST",
      headers: {
        "Content-Type": audioBlob.type || "audio/webm",
        "X-Session-ID": sessionId,
        "X-Turn-ID": turnId,
        "X-Barge-In": String(wasBargeIn),
      },
      body: audioBlob,
    });
    const payload = await response.json();
    pending.remove();
    if (!response.ok) throw new Error(payload.error || `Voice request failed: ${response.status}`);

    if (payload.ignored) {
      voicePlaceholder.remove();
      appendRuntimeEvent(`audio.suppressed | ${payload.ignoreReason}`);
      renderTrace(payload.trace);
      renderSources([]);
      agentBusy = false;
      setListeningState("Listening", "Playback echo was suppressed. Continue speaking naturally.");
      return;
    }

    voicePlaceholder.remove();
    addTranscript("caller", payload.transcript, `STT: ${payload.sttModel}`);
    const ttsMeta = payload.ttsBackend === "provider"
      ? `TTS: ${payload.ttsVoice || payload.ttsModel}`
      : "Browser TTS";
    const meta = [payload.language?.toUpperCase(), ttsMeta, payload.action ? `action: ${payload.action}` : ""]
      .filter(Boolean)
      .join(" | ");
    addTranscript("agent", payload.reply, meta);
    providerEl.textContent = `Provider: ${payload.provider} | ${payload.model} | ${ttsMeta}`;
    languageEl.textContent = payload.language === "es" ? "Spanish" : "English";
    setPipelineMode(payload.pipeline);
    renderSources(payload.sources);
    renderTrace(payload.trace);
    agentBusy = false;
    speak(
      payload.reply,
      payload.locale || "en-US",
      payload.audioBase64 || "",
      payload.audioContentType || "audio/wav",
    );
    if (payload.action === "transfer") agentStatus.textContent = "Transferring";
    if (payload.action === "hangup") agentStatus.textContent = "Call complete";
  } catch (error) {
    pending.remove();
    voicePlaceholder.remove();
    addTranscript("agent", error.message, "error");
    agentBusy = false;
    setListeningState("Error", "The turn failed. Speak again to retry.");
  }
}

function vadLoop() {
  if (!listenStream) return;
  const now = Date.now();
  const rawLevel = audioLevel();
  smoothedLevel = (smoothedLevel * 0.72) + (rawLevel * 0.28);
  const limit = thresholds();

  // Feed the already-EMA-smoothed level to the orb so "listening" scales with the
  // caller's voice. Only written while actually listening (the sole state that
  // consumes --level), avoiding a per-frame DOM write the rest of the time.
  if (voiceOrb && voiceOrb.dataset.voiceState === "listening") {
    voiceOrb.style.setProperty(
      "--level", Math.min(1, smoothedLevel / (limit.start || 0.02)).toFixed(3));
  }

  if (!recorder && !agentSpeaking && !agentBusy && smoothedLevel < limit.start) {
    noiseFloor = (noiseFloor * 0.985) + (rawLevel * 0.015);
  }
  vadReadout.textContent = agentSpeaking
    ? `echo ${playbackEchoFloor.toFixed(3)} | barge ${limit.barge.toFixed(3)}`
    : `noise ${noiseFloor.toFixed(3)} | trigger ${limit.start.toFixed(3)}`;

  if (agentSpeaking && !muted) {
    const playbackAge = now - playbackStartedAt;
    if (playbackAge < tuning.bargeInArmMs) {
      playbackEchoFloor = (playbackEchoFloor * 0.88) + (smoothedLevel * 0.12);
      bargeCandidateAt = 0;
    } else if (smoothedLevel > limit.barge) {
      if (!bargeCandidateAt) {
        bargeCandidateAt = now;
        bargeRecordingCandidate = true;
        appendRuntimeEvent("barge_in.candidate");
        startTurnRecording(true);
        lastSpeechAt = now;
      }
      if (now - bargeCandidateAt >= tuning.bargeInConfirmationMs) {
        bargeRecordingCandidate = false;
        interruptAgent(bargeCandidateAt, true);
        pendingBargeInTurn = false;
        lastSpeechAt = now;
        bargeCandidateAt = 0;
      }
    } else {
      if (bargeRecordingCandidate) {
        stopTurnRecording(true);
        bargeRecordingCandidate = false;
      }
      bargeCandidateAt = 0;
      playbackEchoFloor = (playbackEchoFloor * 0.995) + (smoothedLevel * 0.005);
    }
  } else if (!agentBusy && !muted && now > listenCooldownUntil) {
    if (!recorder) {
      if (smoothedLevel > limit.start) {
        speechCandidateAt = speechCandidateAt || now;
        if (now - speechCandidateAt >= tuning.speechConfirmationMs) {
          startTurnRecording();
          speechCandidateAt = 0;
        }
      } else {
        speechCandidateAt = 0;
      }
    } else {
      if (smoothedLevel > limit.end) lastSpeechAt = now;
      const duration = now - recordingStartedAt;
      const endpointReached = duration >= tuning.minTurnMs
        && now - lastSpeechAt >= tuning.endpointSilenceMs;
      if (endpointReached || duration >= tuning.maxTurnMs) {
        lastEndpointAt = Date.now();
        appendRuntimeEvent(endpointReached ? "vad.endpoint_detected" : "vad.max_turn_reached");
        stopTurnRecording();
      }
    }
  }

  vadFrame = requestAnimationFrame(vadLoop);
}

function attachRoomEvents(room) {
  room.on(RoomEvent.ParticipantConnected, renderParticipants);
  room.on(RoomEvent.ParticipantDisconnected, renderParticipants);
  room.on(RoomEvent.TrackPublished, renderParticipants);
  room.on(RoomEvent.TrackUnpublished, renderParticipants);
  room.on(RoomEvent.Disconnected, renderParticipants);
}

async function connectParticipant(identity, name) {
  const params = new URLSearchParams({ identity, name });
  const response = await fetch(`/token?${params}`);
  if (!response.ok) throw new Error(`Token request failed: ${response.status}`);
  const session = await response.json();
  const room = new Room({ adaptiveStream: true, dynacast: true });
  attachRoomEvents(room);
  await room.connect(session.url, session.token);
  return room;
}

function renderParticipants() {
  participantsEl.innerHTML = "";
  if (!callerRoom) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Participants join when the call starts.";
    participantsEl.appendChild(empty);
    return;
  }

  const participants = [callerRoom.localParticipant, ...callerRoom.remoteParticipants.values()];
  for (const participant of participants) {
    const row = document.createElement("div");
    row.className = "participant";
    const name = document.createElement("strong");
    name.textContent = participant.name || participant.identity;
    const state = document.createElement("span");
    const audioPublished = [...participant.trackPublications.values()]
      .some((publication) => publication.kind === "audio");
    state.textContent = audioPublished ? "audio published" : "room participant";
    row.append(name, state);
    participantsEl.appendChild(row);
  }
}

async function prepareListener() {
  listenStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(listenStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  setListeningState("Calibrating", "Measuring the room noise floor.");
  vadFrame = requestAnimationFrame(vadLoop);
  await new Promise((resolve) => setTimeout(resolve, 650));
  setListeningState("Listening", "Speak naturally. Vera can be interrupted while talking.");
}

async function startCall() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    throw new Error("This browser does not support the required audio APIs.");
  }
  setCallControls(true);
  agentBusy = true;
  callerStatus.textContent = "Connecting";
  agentStatus.textContent = "Connecting";
  await fetch("/reset", { method: "POST", headers: { "X-Session-ID": sessionId } });
  agentRoom = await connectParticipant("vera-agent", "Vera Agent");
  agentStatus.textContent = "Connected";
  await prepareListener();
  callerRoom = await connectParticipant("caller-demo", "Caller Demo");
  await callerRoom.localParticipant.publishTrack(listenStream.getAudioTracks()[0], {
    source: Track.Source.Microphone,
    name: "caller-microphone",
  });
  callerStatus.textContent = "Connected";
  renderParticipants();
  try {
    const greetingResponse = await fetch("/greeting", {
      method: "POST",
      headers: { "X-Session-ID": sessionId },
    });
    const greeting = await greetingResponse.json();
    if (!greetingResponse.ok) throw new Error(greeting.error || "Greeting failed");
    const ttsMeta = greeting.ttsBackend === "provider"
      ? `TTS: ${greeting.ttsVoice || greeting.ttsModel}`
      : "Browser TTS";
    providerEl.textContent = `Provider: ${greeting.provider} | ${greeting.model} | ${ttsMeta}`;
    setPipelineMode(greeting.pipeline);
    renderTrace(greeting.trace);
    agentBusy = false;
    speak(
      greeting.reply,
      greeting.locale || "en-US",
      greeting.audioBase64 || "",
      greeting.audioContentType || "audio/wav",
    );
  } catch (error) {
    agentBusy = false;
    appendRuntimeEvent("tts.greeting_fallback | browser");
    speak("Thanks for calling Vera Hotel reservations. How can I help?", "en-US");
  }
}

async function endCall() {
  if (vadFrame) cancelAnimationFrame(vadFrame);
  vadFrame = null;
  stopTurnRecording(true);
  stopAgentPlayback();
  agentSpeaking = false;
  agentBusy = false;
  bargeRecordingCandidate = false;
  bargeCandidateAt = 0;
  listenStream?.getTracks().forEach((track) => track.stop());
  listenStream = null;
  if (audioContext) await audioContext.close();
  audioContext = null;
  analyser = null;
  callerRoom?.disconnect();
  agentRoom?.disconnect();
  callerRoom = null;
  agentRoom = null;
  callerRoot.classList.remove("speaking");
  agentRoot.classList.remove("speaking");
  callerStatus.textContent = "Ready";
  agentStatus.textContent = "Waiting";
  setListeningState("Idle", "Start the call, then speak naturally");
  setCallControls(false);
  renderParticipants();
}

async function toggleMute() {
  muted = !muted;
  listenStream?.getAudioTracks().forEach((track) => { track.enabled = !muted; });
  muteButton.textContent = muted ? "Unmute" : "Mute";
  callerStatus.textContent = muted ? "Muted" : "Connected";
  setListeningState(muted ? "Muted" : "Listening", muted
    ? "Microphone input is paused."
    : "Speak naturally. Vera can be interrupted while talking.");
}

async function loadState() {
  try {
    const response = await fetch("/state");
    const state = await response.json();
    providerEl.textContent = `Provider: ${state.agentProvider}`;
  } catch {
    providerEl.textContent = "Provider: unavailable";
  }
}

endpointControl.addEventListener("input", () => {
  tuning.endpointSilenceMs = Number(endpointControl.value);
  endpointValue.textContent = `${tuning.endpointSilenceMs} ms`;
});

sensitivityControl.addEventListener("input", () => {
  tuning.sensitivity = Number(sensitivityControl.value);
  sensitivityValue.textContent = `${tuning.sensitivity.toFixed(1)}x`;
});

startButton.addEventListener("click", () => {
  startCall().catch(async (error) => {
    setListeningState("Connection failed", error.message);
    await endCall();
  });
});
muteButton.addEventListener("click", () => toggleMute().catch((error) => {
  setListeningState("Mute failed", error.message);
}));
endButton.addEventListener("click", () => endCall());

setCallControls(false);
loadState();
