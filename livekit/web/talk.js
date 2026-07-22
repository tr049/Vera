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
const callerWaveEl = document.querySelector(".caller-wave");

// Session card shows the provider label only (model/voice live in each bubble's meta).
function renderSession(provider) {
  providerEl.textContent = provider || "—";
}
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
  tts: document.querySelector("#metric-tts"),
  total: document.querySelector("#metric-total"),
  firstAudio: document.querySelector("#metric-first-audio"),
  barge: document.querySelector("#metric-barge"),
};

// A fresh turn clears the panel to a pending state ("—") so stale numbers from the
// previous turn are never mistaken for this one. First audio fills MID-stream at the
// first audible chunk; the rest fill from the trace when the turn completes. The
// barge tile is preserved on a barge-in turn (it was stamped just before this turn).
function resetTurnMetrics(preserveBarge = false) {
  for (const [key, el] of Object.entries(metrics)) {
    if (preserveBarge && key === "barge") continue;
    setMetricValue(el, "—");
  }
  // Fresh turn: the pipeline rail restarts too. VAD relights immediately -- the
  // endpoint that started this turn IS the VAD stage completing.
  for (const stage of document.querySelectorAll("#pipeline [data-stage]")) {
    stage.classList.toggle("complete", stage.dataset.stage === "vad");
    stage.classList.toggle("active", stage.dataset.stage === "stt"); // STT is now in flight
    stage.classList.remove("passed");
  }
}

// Streaming events carry the server's running `timings` snapshot; fill whatever
// stages have finished so the panel ticks along with the turn instead of jumping
// all at once when `final` lands. Server total stays for the final trace.
// Wipe all transient pipeline stage state (used on failure cleanups so a pulsing
// .active node can't linger after a truncated/aborted turn until the next one).
function clearPipelineState() {
  for (const stage of document.querySelectorAll("#pipeline [data-stage]")) {
    stage.classList.remove("active", "complete", "passed");
  }
}

function syncPipelinePassed() {
  const stages = [...document.querySelectorAll("#pipeline [data-stage]")];
  const lastComplete = stages.reduce(
    (last, stage, index) => (stage.classList.contains("complete") ? index : last), -1);
  stages.forEach((stage, index) => {
    stage.classList.toggle("passed",
      index < lastComplete && !stage.classList.contains("complete"));
  });
}

function applyPartialTimings(timings, inFlightHint) {
  if (!timings) return;
  const tiles = { stt: metrics.stt, llm: metrics.llm, tools: metrics.tools, tts: metrics.tts };
  for (const [key, el] of Object.entries(tiles)) {
    if (timings[key] !== undefined) setMetricValue(el, formatMs(timings[key]));
  }
  // inFlightHint (e.g. "tools" at a tool boundary) beats the guess -- otherwise an
  // un-timed Tools stage would light TTS active while the tool is still running.
  const inFlight = inFlightHint
    || ["stt", "llm", "tts"].find((key) => timings[key] === undefined);
  for (const stage of document.querySelectorAll("#pipeline [data-stage]")) {
    if (timings[stage.dataset.stage] !== undefined) stage.classList.add("complete");
    stage.classList.toggle("active", stage.dataset.stage === inFlight);
  }
  syncPipelinePassed();
}

const voiceOrb = document.querySelector(".voice-orb");

// Single chokepoint for every metric write: numeric values count up to their
// target (Framer-style), the row flashes once, and identical rewrites no-op.
// Replaces the old MutationObserver (a count-up would have re-triggered it
// every frame). Reduced-motion: values just appear.
const countupTimers = new WeakMap();

// Typeset "2108 ms" as a mono numeral + a quiet small unit.
function writeMetricText(el, text) {
  let num = el.querySelector(".num");
  if (!num) {
    el.textContent = "";
    num = document.createElement("span");
    num.className = "num";
    const unit = document.createElement("span");
    unit.className = "unit";
    el.append(num, unit);
  }
  const match = /^(\d+) ms$/.exec(text);
  num.textContent = match ? match[1] : text;
  el.querySelector(".unit").textContent = match ? "ms" : "";
}

function setMetricValue(el, text) {
  if (!el || (el.dataset.value === text && !countupTimers.has(el))) return;
  el.dataset.value = text;
  const prev = countupTimers.get(el);
  if (prev) cancelAnimationFrame(prev);
  countupTimers.delete(el);
  const row = el.closest("div");
  if (row) {
    row.classList.remove("bump");
    void row.offsetWidth; // reflow so the flash restarts on every change
    row.classList.add("bump");
  }
  const match = REDUCED_MOTION ? null : /^(\d+) ms$/.exec(text);
  const target = match ? Number(match[1]) : NaN;
  if (!match || !(target > 8)) {
    writeMetricText(el, text);
    return;
  }
  const started = performance.now();
  const duration = Math.min(650, 260 + target / 40); // bigger numbers roll longer
  const tick = (now) => {
    const t = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    writeMetricText(el, `${Math.round(target * eased)} ms`);
    if (t < 1) {
      countupTimers.set(el, requestAnimationFrame(tick));
    } else {
      countupTimers.delete(el);
    }
  };
  countupTimers.set(el, requestAnimationFrame(tick));
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
      animate(".reveal", { y: [18, 0] },
        { delay: stagger(0.05), duration: 0.5, ease: [0.05, 0.7, 0.1, 1] });
      window.setTimeout(() => {
        document.querySelectorAll(".reveal").forEach((el) =>
          el.getAnimations().forEach((a) => a.cancel()));
      }, 1200);
    })
    .catch(() => {}); // Motion unavailable -> no spring polish; the app still works
}

// Magnetic buttons: the call controls lean gently toward the cursor (max ~3px)
// and spring back on leave. Pure transform, pointer-only, reduced-motion off.
if (!REDUCED_MOTION && window.matchMedia("(pointer: fine)").matches) {
  for (const btn of document.querySelectorAll(".call-controls button")) {
    btn.addEventListener("pointermove", (event) => {
      const box = btn.getBoundingClientRect();
      const x = ((event.clientX - box.left) / box.width - 0.5) * 6;
      const y = ((event.clientY - box.top) / box.height - 0.5) * 4;
      btn.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    });
    btn.addEventListener("pointerleave", () => {
      btn.style.transform = "";
    });
  }
}

// Pointer parallax: the orb scene drifts gently toward the cursor over the
// stage (max ~7px), springing back on leave. Transform-only, pointer-only.
{
  const stageEl = document.querySelector(".stage");
  const sceneEl = document.querySelector(".stage-scene");
  if (!REDUCED_MOTION && stageEl && sceneEl
      && window.matchMedia("(pointer: fine)").matches) {
    stageEl.addEventListener("pointermove", (event) => {
      const box = stageEl.getBoundingClientRect();
      const x = ((event.clientX - box.left) / box.width - 0.5) * 14;
      const y = ((event.clientY - box.top) / box.height - 0.5) * 10;
      sceneEl.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    });
    stageEl.addEventListener("pointerleave", () => {
      sceneEl.style.transform = "";
    });
  }
}

// Cursor-following glow on the glass panels (CSS reads --mx/--my).
if (!REDUCED_MOTION && window.matchMedia("(pointer: fine)").matches) {
  for (const panel of document.querySelectorAll(".ops-card, .conversation-column, .side")) {
    panel.addEventListener("pointermove", (event) => {
      const box = panel.getBoundingClientRect();
      panel.style.setProperty("--mx", `${(event.clientX - box.left).toFixed(0)}px`);
      panel.style.setProperty("--my", `${(event.clientY - box.top).toFixed(0)}px`);
    });
  }
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
let lastEndpointAt = 0;
let playbackStartedAt = 0;
let playbackEchoFloor = 0.012;
let pendingBargeInTurn = false;
let currentTurnWasBargeIn = false;
let activeAgentAudio = null;
let playbackToken = 0;
// Streaming-turn state: the in-flight /voice-agent fetch (aborted on barge-in /
// end-call) and the sequential sentence-audio queue it feeds.
let activeTurnAbort = null;
const agentQueue = {
  items: [], playing: false, started: false, streamDone: false, playedAny: false,
  fallbackText: "", fallbackLocale: "en-US",
};

function resetAgentQueue() {
  agentQueue.items.length = 0;
  agentQueue.playing = false;
  agentQueue.started = false;
  agentQueue.streamDone = false;
  agentQueue.playedAny = false;
  agentQueue.fallbackText = "";
  agentQueue.fallbackLocale = "en-US";
}
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
  const active = document.activeElement;
  startButton.disabled = connected;
  muteButton.disabled = !connected;
  endButton.disabled = !connected;
  callerRoot.classList.toggle("connected", connected);
  agentRoot.classList.toggle("connected", connected);
  // Keep keyboard focus on an enabled control when the focused one gets disabled
  // (activating Start would otherwise drop focus to <body>). No focus-steal on init.
  if (connected && active === startButton) muteButton.focus();
  else if (!connected && (active === muteButton || active === endButton)) startButton.focus();
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
  listeningStateEl.dataset.state = VOICE_ORB_STATE[state] || "idle"; // party colour on the pill
  voiceStatusEl.textContent = detail;
  if (voiceOrb) voiceOrb.dataset.voiceState = VOICE_ORB_STATE[state] || "idle";
}

// Landmark trace rows (first audio, barge-in, endpoint) carry the console accent;
// the rest of the timestamp gutter stays subordinate ink.
function isLandmarkEvent(name) {
  return /first_audio|barge|endpoint/.test(name);
}

function setBubbleLabel(labelEl, who, meta) {
  labelEl.textContent = "";
  const speaker = document.createElement("span");
  speaker.textContent = who;
  labelEl.appendChild(speaker);
  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    labelEl.appendChild(metaEl);
  }
}

function addTranscript(role, text, meta = "") {
  transcriptEl.querySelector(".empty")?.remove();
  const item = document.createElement("div");
  item.className = `bubble ${role}`;
  const label = document.createElement("div");
  label.className = "bubble-label";
  setBubbleLabel(label, role === "caller" ? "Caller Demo" : "Vera Agent", meta);
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
  const timings = trace.timings || {};
  setMetricValue(metrics.stt, formatMs(timings.stt));
  setMetricValue(metrics.llm, formatMs(timings.llm));
  setMetricValue(metrics.tools, formatMs(timings.tools));
  // Total synthesis time -- in streaming this is the SUM of every per-sentence
  // synthesize() span (telemetry accumulates same-name spans), not just the first.
  setMetricValue(metrics.tts, formatMs(timings.tts));
  setMetricValue(metrics.total, formatMs(trace.totalMs));

  for (const element of pipelineEl.querySelectorAll("[data-stage]")) {
    const stage = element.dataset.stage;
    const completed = stage === "vad" || timings[stage] !== undefined;
    element.classList.toggle("complete", completed);
    element.classList.remove("active"); // turn finished -- nothing is in flight
  }
  syncPipelinePassed();

  eventsEl.innerHTML = "";
  for (const event of (trace.events || []).slice(-14)) {
    const row = document.createElement("div");
    row.className = isLandmarkEvent(event.name) ? "event-row landmark" : "event-row";
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
  row.className = isLandmarkEvent(name) ? "event-row landmark" : "event-row";
  const time = document.createElement("time");
  time.textContent = "client";
  const detail = document.createElement("span");
  detail.textContent = name;
  row.append(time, detail);
  eventsEl.appendChild(row);
  eventsEl.scrollTop = eventsEl.scrollHeight;
}

function renderSources(sources) {
  sourcesEl.textContent = "";
  if (!sources?.length) {
    sourcesEl.textContent = "No retrieval used in the latest turn.";
    return;
  }
  for (const source of sources) {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = source;
    sourcesEl.appendChild(chip);
  }
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
  resetAgentQueue(); // any queued streamed sentences are stale now
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

// --- Streaming sentence queue: sequential playback of per-sentence WAV chunks.
// Reuses the exact begin/finish/token machinery of the batch path; barge-in
// clears the queue via stopAgentPlayback and aborts the fetch upstream.

function enqueueSentence(item, token) {
  if (token !== playbackToken) return; // stale chunk from an interrupted turn
  agentQueue.items.push(item);
  if (!agentQueue.playing) playNextSentence(token);
}

function maybeFinishQueue(token) {
  // Queue drained AND the stream ended: close the turn. If audio chunks existed
  // but NONE ever played (e.g. autoplay rejection), fall back to the browser
  // voice so the turn is never silent.
  if (token !== playbackToken || agentQueue.playing || !agentQueue.streamDone) return;
  if (!agentQueue.playedAny && agentQueue.fallbackText) {
    const text = agentQueue.fallbackText;
    agentQueue.fallbackText = "";
    appendRuntimeEvent("tts.stream_playback_failed | browser fallback");
    speakWithBrowserVoice(text, agentQueue.fallbackLocale, token);
    return;
  }
  if (agentQueue.started) finishAgentPlayback(token);
}

function playNextSentence(token) {
  if (token !== playbackToken) return;
  const item = agentQueue.items.shift();
  if (!item) {
    agentQueue.playing = false;
    maybeFinishQueue(token);
    return;
  }
  if (!item.audioBase64) {
    playNextSentence(token); // TTS failed for this sentence -> skip it
    return;
  }
  agentQueue.playing = true;
  const audio = new Audio(`data:${item.audioContentType || "audio/wav"};base64,${item.audioBase64}`);
  activeAgentAudio = audio;
  const advance = () => {
    if (token !== playbackToken) return;
    playNextSentence(token);
  };
  audio.onplay = () => {
    if (token !== playbackToken) return;
    agentQueue.playedAny = true;
    if (!agentQueue.started) {
      agentQueue.started = true;
      beginAgentPlayback(token, "provider"); // arms barge-in + first_audio ONCE
      agentBusy = false; // barge-in is live while the rest still streams
    }
  };
  audio.onended = advance;
  audio.onerror = advance;
  audio.play().catch(advance);
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
    setMetricValue(metrics.firstAudio, formatMs(firstAudioMs));
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
  activeTurnAbort?.abort(); // close the stream: server stops synthesizing
  stopAgentPlayback();
  agentSpeaking = false;
  agentRoot.classList.remove("speaking");
  listenCooldownUntil = Date.now() + 80;
  pendingBargeInTurn = !turnAlreadyRecording;
  addInterruption();
  appendRuntimeEvent("barge_in.detected");
  setMetricValue(metrics.barge, formatMs(Date.now() - detectedAt));
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

function renderIgnoredTurn(payload, refs) {
  refs.voicePlaceholder?.remove();
  refs.callerBubble?.remove(); // stream_error path: a transcript event already consumed the placeholder
  refs.pending?.remove();
  appendRuntimeEvent(`audio.suppressed | ${payload.ignoreReason}`);
  renderTrace(payload.trace);
  // Nothing played on a suppressed turn -- clear First audio so the row isn't this
  // turn's STT/LLM next to a PRIOR turn's first-audio latency.
  setMetricValue(metrics.firstAudio, "—");
  renderSources([]);
  agentBusy = false;
  setListeningState("Listening", "Playback echo was suppressed. Continue speaking naturally.");
}

function renderFinalTurn(payload, refs) {
  // Shared final rendering for batch AND streaming turns (transcript bubble may
  // already exist on the streaming path -- refs are nulled once consumed).
  refs.pending?.remove();
  refs.voicePlaceholder?.remove();
  if (!refs.transcriptShown && payload.transcript) {
    addTranscript("caller", payload.transcript, `STT: ${payload.sttModel}`);
  }
  const ttsMeta = payload.ttsBackend === "provider"
    ? `TTS: ${payload.ttsVoice || payload.ttsModel}`
    : "Browser TTS";
  const meta = [payload.language?.toUpperCase(), ttsMeta, payload.action ? `action: ${payload.action}` : ""]
    .filter(Boolean)
    .join(" | ");
  addTranscript("agent", payload.reply, meta);
  renderSession(payload.provider);
  languageEl.textContent = payload.language === "es" ? "Spanish" : "English";
  setPipelineMode(payload.pipeline);
  renderSources(payload.sources);
  renderTrace(payload.trace);
  if (payload.action === "transfer") agentStatus.textContent = "Transferring";
  if (payload.action === "hangup") agentStatus.textContent = "Call complete";
}

async function consumeVoiceStream(response, refs) {
  // NDJSON events: transcript -> (sentence | reset)* -> final. The connection
  // close ends the stream; a missing `final` means the turn was truncated.
  stopAgentPlayback(); // fresh token for this turn's sentence queue
  const token = playbackToken;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  let sawFinal = false;
  refs.streamedText = ""; // on refs so the abort path can preserve heard text

  const handleEvent = (event) => {
    if (event.type === "transcript") {
      applyPartialTimings(event.timings); // STT just completed server-side
      // Fill the caller placeholder IN PLACE. Removing it and re-adding via
      // addTranscript() would append the real transcript at the BOTTOM of the
      // thread -- below the agent "Processing turn" bubble that already exists --
      // so the reply would render above the caller until `final` reordered them.
      refs.transcriptShown = true;
      if (refs.voicePlaceholder) {
        setBubbleLabel(refs.voicePlaceholder.querySelector(".bubble-label"),
          "Caller Demo", `STT: ${event.sttModel}`);
        refs.voicePlaceholder.querySelector("div:last-child").textContent = event.transcript;
        refs.voicePlaceholder.removeAttribute("aria-hidden"); // real transcript -> announce it
        // Keep a handle to the now-real caller bubble: voicePlaceholder is nulled so
        // the truncation/abort cleanups leave it alone, but an ignored `final`
        // (stream_error, which arrives AFTER a transcript) must still remove it.
        refs.callerBubble = refs.voicePlaceholder;
        refs.voicePlaceholder = null; // consumed: it is now the real caller bubble
      } else {
        addTranscript("caller", event.transcript, `STT: ${event.sttModel}`);
      }
    } else if (event.type === "sentence") {
      applyPartialTimings(event.timings); // tts accumulates; llm/tools once closed
      refs.streamedText = refs.streamedText ? `${refs.streamedText} ${event.text}` : event.text;
      if (refs.pending) {
        refs.pending.querySelector("div:last-child").textContent = refs.streamedText;
      }
      enqueueSentence(
        { audioBase64: event.audioBase64, audioContentType: event.audioContentType },
        token,
      );
    } else if (event.type === "reset") {
      applyPartialTimings(event.timings, "tools"); // tool boundary: Tools is now in flight
      // Tool boundary/degrade: queued-but-unplayed preamble audio is stale. Reset the
      // playback-progress flags too so the post-tool reply is treated as a FRESH segment
      // -- otherwise a played preamble leaves playedAny=true and the real reply's
      // silent-turn (all-TTS-failed) browser-voice fallback never fires.
      refs.streamedText = "";
      agentQueue.items.length = 0;
      agentQueue.playedAny = false;
      agentQueue.started = false;
      if (refs.pending) {
        refs.pending.querySelector("div:last-child").textContent = "Processing turn";
      }
    } else if (event.type === "final") {
      sawFinal = true;
      agentQueue.streamDone = true;
      if (event.ignored) {
        renderIgnoredTurn(event, refs);
        return;
      }
      renderFinalTurn(event, refs);
      refs.pending = null;
      agentBusy = false;
      if (event.ttsFallback && !agentQueue.playedAny) {
        // No audio chunk made it -- speak the whole reply with the browser voice.
        speak(event.reply, event.locale || "en-US", "", "audio/wav");
        return;
      }
      // If chunks exist but none ever PLAYS (autoplay rejection), the drain
      // check speaks the reply with the browser voice instead of going silent.
      agentQueue.fallbackText = event.reply || "";
      agentQueue.fallbackLocale = event.locale || "en-US";
      maybeFinishQueue(token);
    } else if (event.type === "error") {
      appendRuntimeEvent(`stream.error | ${event.error}`);
    }
  };

  const STREAM_IDLE_MS = 20000; // a half-open stream that stalls without closing must not
  while (true) {                // freeze the turn loop -- abort it so cleanup runs.
    let idleTimer;
    const idle = new Promise((resolve) => {
      idleTimer = setTimeout(() => resolve("__idle__"), STREAM_IDLE_MS);
    });
    const readPromise = reader.read();
    readPromise.catch(() => {}); // swallow the post-abort rejection (avoid unhandled)
    const result = await Promise.race([readPromise, idle]);
    clearTimeout(idleTimer);
    if (result === "__idle__") {
      appendRuntimeEvent("stream.idle_timeout");
      activeTurnAbort?.abort(); // -> falls through to the !sawFinal cleanup below
      break;
    }
    const { done, value } = result;
    if (done) break;
    buffered += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffered.indexOf("\n")) >= 0) {
      const line = buffered.slice(0, newline).trim();
      buffered = buffered.slice(newline + 1);
      if (line) handleEvent(JSON.parse(line));
    }
  }
  if (!sawFinal) {
    appendRuntimeEvent("stream.truncated");
    refs.pending?.remove();
    refs.voicePlaceholder?.remove();
    clearPipelineState();
    agentBusy = false;
    // Close out playback state so agentSpeaking can't stay stuck true (which
    // would leave the VAD in barge-mode forever).
    agentQueue.streamDone = true;
    agentQueue.fallbackText = "";
    if (agentQueue.playing) {
      agentQueue.items.length = 0; // let the current sentence finish, then close
    } else {
      maybeFinishQueue(token);
    }
    if (listenStream) {
      setListeningState("Listening", "Speak naturally. Vera can be interrupted while talking.");
    }
  }
}

async function sendAudioToAgent(audioBlob) {
  agentBusy = true;
  setListeningState("Processing", "Transcribing and running the hotel agent.");
  const refs = {
    voicePlaceholder: addTranscript("caller", "Voice turn", "transcribing"),
    pending: addTranscript("agent", "Processing turn", "STT -> Router -> RAG -> LLM -> Tools"),
    transcriptShown: false,
  };
  // Placeholders are transient scaffolding -- hide them from the aria-live transcript log
  // (the role=status listening-state already announces turn progress). Un-hidden once real.
  refs.voicePlaceholder.setAttribute("aria-hidden", "true");
  refs.pending.setAttribute("aria-hidden", "true");
  const turnId = `turn-${++turnCounter}`;
  const wasBargeIn = currentTurnWasBargeIn;
  currentTurnWasBargeIn = false;
  // Fresh turn -> clear the telemetry panel now, not when the turn completes.
  resetTurnMetrics(wasBargeIn);
  activeTurnAbort = new AbortController();

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
      signal: activeTurnAbort.signal,
    });

    if ((response.headers.get("Content-Type") || "").includes("application/x-ndjson")) {
      await consumeVoiceStream(response, refs);
      return;
    }

    const payload = await response.json();
    refs.pending.remove();
    refs.pending = null;
    if (!response.ok) throw new Error(payload.error || `Voice request failed: ${response.status}`);

    if (payload.ignored) {
      renderIgnoredTurn(payload, refs);
      return;
    }

    refs.voicePlaceholder.remove();
    refs.voicePlaceholder = null;
    renderFinalTurn(payload, refs);
    agentBusy = false;
    speak(
      payload.reply,
      payload.locale || "en-US",
      payload.audioBase64 || "",
      payload.audioContentType || "audio/wav",
    );
  } catch (error) {
    if (error.name === "AbortError") {
      // Intentional: barge-in or end-call cancelled the turn mid-stream. Keep
      // the sentences the caller already HEARD in the transcript instead of
      // erasing them (the agent's history keeps the full reply server-side).
      if (refs.pending && refs.streamedText) {
        const label = refs.pending.querySelector(".bubble-label");
        const flag = document.createElement("span");
        flag.className = "meta";
        flag.textContent = "interrupted";
        label.appendChild(flag); // preserves the speaker/meta span structure
        refs.pending = null;
      }
      clearPipelineState();
      refs.pending?.remove();
      refs.voicePlaceholder?.remove();
      appendRuntimeEvent("turn.aborted");
      agentBusy = false;
      return;
    }
    refs.pending?.remove();
    refs.voicePlaceholder?.remove();
    // A mid-stream network failure can leave sentence audio queued/playing --
    // close it all out so agentSpeaking can't stay stuck true.
    stopAgentPlayback();
    agentSpeaking = false;
    agentRoot.classList.remove("speaking");
    addTranscript("agent", error.message, "error").classList.add("error");
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

  // Live mic meter: written every frame while the call is up (a real-time input
  // indicator, so it stays honest even during agent playback) -- one compositor
  // custom-property write, no layout.
  const micLevel = Math.min(1, smoothedLevel / (limit.start || 0.02)).toFixed(3);
  if (callerWaveEl) {
    callerWaveEl.style.setProperty("--mic", micLevel);
  }
  // The orb only consumes --level while "listening", so write it only then.
  if (voiceOrb && voiceOrb.dataset.voiceState === "listening") {
    voiceOrb.style.setProperty("--level", micLevel);
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
    const party = /caller/i.test(participant.identity || participant.name || "") ? "caller" : "agent";
    row.className = `participant ${party}`;
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
  // Fresh call: clear per-turn latency state so the greeting/first turn can't display a
  // prior call's First audio, and a leftover endpoint stamp can't skew the next metric.
  lastEndpointAt = 0;
  for (const el of Object.values(metrics)) setMetricValue(el, "0 ms");
  languageEl.textContent = "English"; // reset agent starts in English; don't show prior call's language
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
    renderSession(greeting.provider);
    setPipelineMode(greeting.pipeline);
    renderTrace(greeting.trace);
    agentBusy = false;
    if (!listenStream) return; // hung up during the in-flight greeting fetch -> stay silent
    addTranscript("agent", greeting.reply, ttsMeta);
    speak(
      greeting.reply,
      greeting.locale || "en-US",
      greeting.audioBase64 || "",
      greeting.audioContentType || "audio/wav",
    );
  } catch (error) {
    agentBusy = false;
    if (!listenStream) return;
    const fallback = "Thanks for calling Vera Hotel reservations. How can I help?";
    appendRuntimeEvent("tts.greeting_fallback | browser");
    addTranscript("agent", fallback, "Browser TTS");
    speak(fallback, "en-US");
  }
}

async function endCall() {
  if (vadFrame) cancelAnimationFrame(vadFrame);
  vadFrame = null;
  stopTurnRecording(true);
  activeTurnAbort?.abort(); // kill any in-flight streamed turn
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
  // Muting mid-turn: discard the in-flight recorder. Its endpoint/maxTurn cutoffs run
  // in vadLoop gated behind !muted, so without this the recorder would run unbounded.
  if (muted) stopTurnRecording(true);
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
    renderSession(state.agentProvider);
  } catch {
    renderSession("unavailable");
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
    // Clean up FIRST -- endCall() ends with setListeningState("Idle", ...), which would
    // otherwise overwrite the error in the same microtask (no paint between them).
    await endCall();
    setListeningState("Connection failed", error.message);
    addTranscript("agent", error.message, "error").classList.add("error");
  });
});
muteButton.addEventListener("click", () => toggleMute().catch((error) => {
  setListeningState("Mute failed", error.message);
}));
endButton.addEventListener("click", () => endCall());

setCallControls(false);
loadState();
