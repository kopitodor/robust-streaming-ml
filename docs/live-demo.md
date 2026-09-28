# Live-microphone demo (local only)

A browser page that listens to your microphone and runs the pinned, parity-verified ONNX export through the existing `kw10-streaming-v1` streaming engine, window by window. It adds to the file-based demo ([demo.md](demo.md)), which is unchanged.

> **Closed-set demo.** It recognises only `yes, no, up, down, left, right, on, off, stop, go`, and every 1 s window gets one of them, including silence, noise, music, and other words. It is not general speech recognition and not production software. There is no silence or unknown-word rejection.

## Launch

Requirements:
- `uv sync`;
- the Milestone 6 export run, a local and git-ignored artifact pinned in [`configs/kw10_demo.yaml`](../configs/kw10_demo.yaml). The server refuses to start without it. See [demo.md](demo.md#what-you-need).

```bash
uv run python scripts/live_demo.py
```

Then open <http://127.0.0.1:8765/> and press **Start listening**.

| Option | Meaning |
| --- | --- |
| `--port N` | listen on another port (`0` picks any free port); default `live.port`, 8765 |
| `--host H` | a loopback address (`127.0.0.1`, `::1`, `localhost`). **Anything else is refused.** |
| `--open` | open the page in your default browser |
| `--config` | defaults to `configs/kw10_demo.yaml` |

Stop the server with Ctrl+C.

## Browser requirements and microphone permission

- **Browser:** a current desktop Chrome, Edge, or Firefox with `getUserMedia` and `AudioWorklet`. Safari was not tested.
- **Secure context:** the page must be opened as `http://127.0.0.1:<port>` or `http://localhost:<port>`. Browsers treat these as secure contexts, which microphone access requires. A LAN address would not qualify, and the server cannot be bound to one anyway.
- **Permission:** nothing is captured until you press **Start listening**. The browser then asks for microphone permission.
  - If you deny it, the page says so. To retry, allow the microphone in the browser's site settings (the icon left of the address bar) and press Start again.
  - **Stop** and **Reset** stop the microphone tracks, so the browser's recording indicator turns off.
- **Audio processing:** the page asks the browser to turn off echo cancellation, noise suppression, and automatic gain. Browsers and operating systems may still process or resample audio before the page receives it, which is outside this project's control.

## Controls and display

- **Start listening:** requests the microphone and begins a new stream.
- **Stop:** ends the stream. The remaining audio is flushed, and the page shows the **final decision**: the `kw10-streaming-v1` stream decision, which is the aggregate of the last window.
- **Reset:** stops capture, discards everything on the server, and clears the page.

For every window, which is 1 s long with one starting every 100 ms, the page shows:
- the label and confidence;
- the time span in seconds from the start of the stream;
- the input level in dBFS;
- the ONNX Runtime inference time;
- the **running aggregate**: the mean of the last 5 windows' probabilities (`kw10-streaming-v1`), with its label, confidence, and how many windows it uses.

The **stable** badge appears only when the aggregate uses the full 5-window span and all 5 windows' top-1 labels equal the aggregate label. This is a display rule; predictions are unchanged. Stable means only that the windows agree. On silence or noise the model can be stable on an arbitrary command at low confidence, so check the confidence and the level. Before 5 windows exist, the badge says "warming up".

## Audio path (documented and deterministic from the page onwards)

1. **Browser.** An `AudioWorklet` (`capture-worklet.js`) receives the microphone at the audio context's rate `R`, usually 48,000 or 44,100 Hz. It averages the channels to mono (arithmetic mean). The page groups the samples into 100 ms chunks and encodes them as little-endian float32 (`f32le`).
2. **Transport.** The chunks are posted in order, with sequence numbers 0, 1, 2, …, to the local server. Each request waits for the previous one to finish.
3. **Server.** Each chunk is validated against the audio contract below. It is then converted to 16 kHz by `robust_streaming_ml.live.resample.StreamingResampler`:
   - a Kaiser-windowed sinc lowpass with cutoff `min(1, 16000/R)`, 16 zero crossings, and `beta = 8`;
   - each phase's taps normalised to unit DC gain;
   - zero samples before the start of the stream, and after its end on Stop;
   - `ceil(N · 16000 / R)` outputs for `N` inputs.

   The output is clipped to `[-1, 32767/32768]` and rounded half to even onto the 16-bit grid, the same PCM format as the dataset. `R = 16000` passes through unchanged. The conversion is bit-identical for any chunking (tested).
4. **Inference.** The 16 kHz PCM is pushed into the existing `StreamingClassifier`, which runs 1 s windows every 100 ms, one ONNX Runtime CPU call per window, and the causal aggregate of the last 5 windows.

## Local API and audio contract

The server uses the standard library only and listens on a loopback address.

| Method | Path | Body | Result |
| --- | --- | --- | --- |
| GET | `/`, `/app.js`, `/capture-worklet.js`, `/style.css` | – | the page |
| GET | `/api/info` | – | model identity, labels, streaming contract, audio contract, stable rule |
| POST | `/api/sessions` | JSON `{"sample_rate_hz": R}` | `201 {"session", "sample_rate_hz", "next_seq": 0}` |
| POST | `/api/sessions/<id>/chunks?seq=<n>` | `application/octet-stream`, `f32le` mono at `R` | the windows this chunk completed, plus stream status |
| POST | `/api/sessions/<id>/stop` | – | the remaining windows and `final_decision` (or `null` for an empty stream) |
| POST | `/api/sessions/<id>/reset` | – | the initial state (`next_seq` 0) |
| DELETE | `/api/sessions/<id>` | – | the session is discarded |

**Contract:**
- `R` must be one of 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, or 96000 Hz.
- A chunk holds 1 to `2·R` samples (at most 2 s), a whole number of float32 values, all finite.
- `seq` must be exactly the next expected number.

**Violations change nothing on the server:**
- 400: malformed audio, a bad `seq`, an unsupported rate, or an invalid JSON body;
- 409: a duplicate or out-of-order chunk, or audio after Stop;
- 413: too large;
- 415: the wrong content type.

**Security:**
- only loopback addresses can be bound;
- requests with a foreign `Host` header (DNS rebinding) and writes with a foreign `Origin` are refused with 403;
- no CORS headers are sent;
- pages carry a strict Content-Security-Policy (`default-src 'self'`, no inline scripts);
- the page loads no third-party resources.

**Privacy:** audio is processed in memory and never written to disk. Nothing is uploaded, and there is no share link. At most 4 sessions are kept; the least recently used is dropped.

## What was tested

**Automated** (`tests/test_live.py`):
- the resampler: output counts for every supported rate; bit-identical output under any chunking; agreement with an independent implementation of the formula; DC gain, passband, and alias rejection; quantisation; invalid input;
- the audio contract;
- chunk ordering (duplicates and gaps rejected with the session unchanged);
- reset (equal to a fresh session);
- Stop semantics and the empty stream;
- the stable rule;
- the HTTP API against a real ONNX export: a browser-contract simulation whose windows equal the offline pipeline exactly; malformed-audio rejection; unsupported rates; unknown sessions; `Host` and `Origin` checks; loopback-only binding; session eviction; the static page's disclaimers and CSP; and that nothing is written to disk.

**Manual** (2026-09-28, in the Claude desktop browser pane, against the real pinned export). This was not a real microphone: `getUserMedia` was replaced in the page by a stream playing real test audio. Everything after that point is the real code path: AudioContext at 48,000 Hz, the capture worklet, chunking, the server, the resampler, and ONNX inference. See RESULTS.md for the observations. **A real microphone was not tested by the developer. Please try it yourself.**

## Limitations

- **Closed set:** 10 commands only, with no silence or unknown-word rejection. It is not general speech recognition and not production software, with no reliability or real-time guarantee.
- **Scope:** the model was trained on isolated 1 s Speech Commands clips, from one model and one seed. Accuracy on your microphone, voice, accent, room, and noise is unknown and was not evaluated.
- **Browser processing:** audio before the page receives it (browser or OS resampling and processing) is outside the documented deterministic path.
- **Timings** are this session's per-window inference times on this machine. They are not the `kw10-latency-v1` benchmark.
- **Stream decision:** after Stop it reflects only the last 5 windows. With several words or trailing silence, the per-window table is the informative view.
