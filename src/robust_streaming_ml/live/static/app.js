"use strict";

// Local live demo client. Captures only after "Start listening" and microphone
// permission; sends ordered 100 ms f32le chunks to this local server; renders the
// windows it returns. No third-party code, no other network destinations.

const CHUNK_SECONDS = 0.1;
const HISTORY_ROWS = 20;
const LIMITATIONS = [
  "Closed set: every window gets one of the 10 commands, including silence, noise, and other words. There is no silence or unknown-word rejection.",
  "Not general speech recognition, and not production software. No reliability or real-time guarantee.",
  "Trained and evaluated on isolated 1 s Speech Commands clips from one model and one seed. Accuracy on your microphone, voice, room, and accent is unknown.",
  "The browser and operating system may still process the audio (resampling to the audio-context rate, device-level processing).",
  "Stable only means that the last 5 windows agree. Silence and noise can also give a stable but meaningless label, so check the confidence and the input level.",
  "Timings are measured in this session on this machine. The protocol benchmark is kw10-latency-v1 in RESULTS.md.",
];

const state = {
  info: null,
  session: null,
  sessionRate: null,
  ctx: null,
  stream: null,
  node: null,
  sink: null,
  pending: [],
  sending: null,
  seq: 0,
  frames: [],
  framed: 0,
  chunkFrames: 0,
  listening: false,
  stopped: false,
  windows: 0,
};

const $ = (id) => document.getElementById(id);

function setStatus(text, kind = "info") {
  const el = $("status");
  el.textContent = text;
  el.dataset.kind = kind;
}

function setButtons(mode) {
  $("start").disabled = mode === "listening" || mode === "busy";
  $("stop").disabled = mode !== "listening";
  $("reset").disabled = mode === "busy" || state.session === null;
}

async function api(path, init = {}) {
  const response = await fetch(path, init);
  let body = {};
  try {
    body = await response.json();
  } catch (_) {
    body = {};
  }
  if (!response.ok) {
    throw new Error(body.error || `${response.status} ${response.statusText}`);
  }
  return body;
}

function text(tag, value, className) {
  const el = document.createElement(tag);
  el.textContent = value;
  if (className) el.className = className;
  return el;
}

async function loadInfo() {
  state.info = await api("/api/info");
  const chips = $("commands");
  chips.replaceChildren(...state.info.labels.map((label) => {
    const li = text("li", label);
    li.dataset.label = label;
    return li;
  }));
  const model = state.info.model || {};
  $("model").textContent = `ONNX model ${String(model.onnx_sha256 || "?").slice(0, 12)}… from export run ${model.export_run_id || "?"} (parity passed: ${model.export_parity_passed}); checkpoint ${String((model.checkpoint || {}).sha256 || "?").slice(0, 12)}…; labels in order: ${state.info.labels.join(", ")}.`;
  $("contract").textContent = `${state.info.audio_contract.conversion}. Supported device rates: ${state.info.audio_contract.supported_device_rates_hz.join(", ")} Hz.`;
  $("stable-rule").textContent = `${state.info.stable_rule}.`;
  $("limitations").replaceChildren(...LIMITATIONS.map((item) => text("li", item)));
  setStatus('Ready. Press "Start listening"; the browser will ask for microphone permission.');
  setButtons("idle");
}

function percent(value) {
  return `${(100 * value).toFixed(1)}%`;
}

function clearOutputs() {
  state.windows = 0;
  $("history").replaceChildren();
  $("current-label").textContent = "–";
  $("current-detail").textContent = "no window yet";
  $("current-bar").style.width = "0";
  $("aggregate-label").textContent = "–";
  $("aggregate-detail").textContent = " ";
  $("aggregate-bar").style.width = "0";
  const badge = $("stability");
  badge.dataset.state = "idle";
  badge.textContent = "no decision yet";
  $("final-card").hidden = true;
  for (const chip of $("commands").children) chip.classList.remove("active");
}

function renderWindow(w) {
  state.windows += 1;
  $("current-label").textContent = w.label;
  $("current-bar").style.width = percent(w.confidence);
  const level = w.level_dbfs === null ? "silent" : `${w.level_dbfs.toFixed(1)} dBFS`;
  $("current-detail").textContent = `confidence ${w.confidence.toFixed(3)} · window #${w.index}, ${w.start_s.toFixed(2)}–${w.end_s.toFixed(2)} s${w.final_partial ? " (zero-padded)" : ""} · level ${level} · inference ${w.inference_ms.toFixed(2)} ms`;
  $("aggregate-label").textContent = w.aggregate_label;
  $("aggregate-bar").style.width = percent(w.aggregate_confidence);
  $("aggregate-detail").textContent = `confidence ${w.aggregate_confidence.toFixed(3)} · mean of ${w.aggregate_windows} of 5 windows`;
  const badge = $("stability");
  if (w.stable) {
    badge.dataset.state = "stable";
    badge.textContent = `stable: all 5 windows say "${w.aggregate_label}" (aggregate confidence ${w.aggregate_confidence.toFixed(3)})`;
  } else if (w.aggregate_windows < 5) {
    badge.dataset.state = "idle";
    badge.textContent = `warming up (${w.aggregate_windows}/5 windows)`;
  } else {
    badge.dataset.state = "idle";
    badge.textContent = "not stable: the last 5 windows disagree";
  }
  for (const chip of $("commands").children) {
    chip.classList.toggle("active", chip.dataset.label === w.aggregate_label);
  }
  const row = document.createElement("tr");
  row.append(
    text("td", String(w.index), "num"),
    text("td", `${w.start_s.toFixed(2)}–${w.end_s.toFixed(2)}`, "num"),
    text("td", w.label),
    text("td", w.confidence.toFixed(3), "num"),
    text("td", w.aggregate_label),
    text("td", w.aggregate_confidence.toFixed(3), "num"),
    text("td", w.stable ? "yes" : "no"),
    text("td", w.level_dbfs === null ? "–" : w.level_dbfs.toFixed(1), "num"),
    text("td", w.inference_ms.toFixed(2), "num"),
  );
  const history = $("history");
  history.prepend(row);
  while (history.children.length > HISTORY_ROWS) history.lastChild.remove();
}

function renderResult(result) {
  for (const w of result.windows) renderWindow(w);
  const backlog = state.pending.length * CHUNK_SECONDS;
  $("counters").textContent = `device rate ${state.sessionRate} Hz · chunks accepted ${result.next_seq ?? state.seq} · audio at 16 kHz ${result.seconds.toFixed(1)} s · windows ${state.windows} · backlog ${backlog.toFixed(1)} s`;
}

function encodeChunk(frames) {
  const buffer = new ArrayBuffer(frames.length * 4);
  const view = new DataView(buffer);
  for (let i = 0; i < frames.length; i += 1) view.setFloat32(i * 4, frames[i], true);
  return buffer;
}

function takeFrames(count) {
  const out = new Float32Array(count);
  let filled = 0;
  while (filled < count) {
    const head = state.frames[0];
    const need = count - filled;
    if (head.length <= need) {
      out.set(head, filled);
      filled += head.length;
      state.frames.shift();
    } else {
      out.set(head.subarray(0, need), filled);
      state.frames[0] = head.subarray(need);
      filled += need;
    }
  }
  state.framed -= count;
  return out;
}

function onFrames(mono) {
  if (!state.listening) return;
  state.frames.push(mono);
  state.framed += mono.length;
  while (state.framed >= state.chunkFrames) {
    state.pending.push(encodeChunk(takeFrames(state.chunkFrames)));
  }
  kick();
}

async function drain() {
  while (state.pending.length > 0) {
    const body = state.pending.shift();
    const seq = state.seq;
    const result = await api(`/api/sessions/${state.session}/chunks?seq=${seq}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body,
    });
    state.seq = seq + 1;
    renderResult(result);
  }
}

function kick() {
  if (state.sending) return;
  state.sending = drain()
    .catch(async (error) => {
      state.pending = [];
      await stopCapture();
      setButtons("idle");
      setStatus(`Stopped: ${error.message}`, "error");
    })
    .finally(() => {
      state.sending = null;
      if (state.pending.length > 0 && state.listening) kick();
    });
}

async function settle() {
  while (state.sending) {
    await state.sending;
  }
}

async function stopCapture() {
  state.listening = false;
  if (state.node) {
    state.node.port.onmessage = null;
    state.node.disconnect();
    state.node = null;
  }
  if (state.sink) {
    state.sink.disconnect();
    state.sink = null;
  }
  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
    state.stream = null;
  }
  if (state.ctx) {
    await state.ctx.close();
    state.ctx = null;
  }
}

async function start() {
  if (state.listening) return;
  const supported = window.isSecureContext && navigator.mediaDevices
    && navigator.mediaDevices.getUserMedia && window.AudioWorkletNode;
  if (!supported) {
    setStatus("This browser cannot capture audio here. The demo needs getUserMedia and AudioWorklet on http://127.0.0.1 or http://localhost; use a current Chrome, Edge, or Firefox.", "error");
    return;
  }
  setButtons("busy");
  try {
    setStatus("Waiting for microphone permission…");
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });
    state.ctx = new AudioContext();
    await state.ctx.audioWorklet.addModule("/capture-worklet.js");
    const rate = state.ctx.sampleRate;
    if (state.session !== null && state.sessionRate === rate) {
      await api(`/api/sessions/${state.session}/reset`, { method: "POST" });
    } else {
      if (state.session !== null) {
        await api(`/api/sessions/${state.session}`, { method: "DELETE" }).catch(() => undefined);
      }
      const created = await api("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_rate_hz: rate }),
      });
      state.session = created.session;
      state.sessionRate = rate;
    }
    state.seq = 0;
    state.stopped = false;
    state.frames = [];
    state.framed = 0;
    state.pending = [];
    state.chunkFrames = Math.round(rate * CHUNK_SECONDS);
    clearOutputs();
    const source = state.ctx.createMediaStreamSource(state.stream);
    state.node = new AudioWorkletNode(state.ctx, "rsml-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    state.sink = state.ctx.createGain();
    state.sink.gain.value = 0; // keeps the graph running without playing the microphone back
    state.node.port.onmessage = (event) => onFrames(event.data);
    source.connect(state.node).connect(state.sink).connect(state.ctx.destination);
    state.listening = true;
    setButtons("listening");
    setStatus(`Listening (device rate ${rate} Hz). The first window appears after 1 s of audio.`);
  } catch (error) {
    await stopCapture();
    setButtons("idle");
    const reason = error && error.name === "NotAllowedError"
      ? "microphone permission was denied; allow it in the browser's site settings and press Start again"
      : error.message || String(error);
    setStatus(`Could not start: ${reason}`, "error");
  }
}

async function stop() {
  if (!state.listening) return;
  setButtons("busy");
  await stopCapture();
  if (state.framed > 0) state.pending.push(encodeChunk(takeFrames(state.framed)));
  kick();
  await settle();
  try {
    const result = await api(`/api/sessions/${state.session}/stop`, { method: "POST" });
    renderResult(result);
    state.stopped = true;
    const final = result.final_decision;
    $("final-card").hidden = false;
    $("final-label").textContent = final ? final.label : "no decision";
    $("final-detail").textContent = final
      ? `confidence ${final.confidence.toFixed(3)} · ${final.rule} · ${final.windows} window(s)`
      : "no audio reached the model";
    const badge = $("stability");
    badge.dataset.state = final ? "final" : "idle";
    badge.textContent = final ? `final: "${final.label}"` : "no decision";
    setStatus("Stopped. The final decision is the aggregate of the last window. Press Start for a new stream, or Reset.");
  } catch (error) {
    setStatus(`Stop failed: ${error.message}`, "error");
  }
  setButtons("idle");
}

async function reset() {
  setButtons("busy");
  await stopCapture();
  state.pending = [];
  state.frames = [];
  state.framed = 0;
  await settle().catch(() => undefined);
  try {
    if (state.session !== null) await api(`/api/sessions/${state.session}/reset`, { method: "POST" });
    state.seq = 0;
    state.stopped = false;
    clearOutputs();
    $("counters").textContent = "";
    setStatus('Reset. Press "Start listening" to begin a new stream.');
  } catch (error) {
    setStatus(`Reset failed: ${error.message}`, "error");
  }
  setButtons("idle");
}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
$("reset").addEventListener("click", reset);
loadInfo().catch((error) => setStatus(`Cannot reach the local server: ${error.message}`, "error"));
