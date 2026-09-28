# Streaming and ONNX protocol: `kw10-streaming-v1` and `kw10-onnx-v1`

Milestone 6. This document was written and committed **before** the streaming and export code, and before any real-data export or parity run. It fixes:
- the streaming contract (`kw10-streaming-v1`): input, window, hop, timestamps, padding, short and final windows, reset, per-window prediction, and aggregation;
- the export and parity protocol (`kw10-onnx-v1`): exported graph, opset, tensor contract, runtime, provider, metadata, and numerical tolerances.

Results will go in [RESULTS.md](../RESULTS.md).

**Implementation** (committed after this document, before any real run):
- `robust_streaming_ml.streaming`: `windowing.py` and `classifier.py`;
- `robust_streaming_ml.export`: `onnx_model.py`, `parity.py`, `config.py`, and `run.py`;
- entry point `scripts/export_onnx.py`, with [`configs/kw10_cnn_baseline_onnx.yaml`](../configs/kw10_cnn_baseline_onnx.yaml).

Nothing here retrains, tunes, or recalibrates the model, and nothing here measures latency (that is Milestone 7).

## 1. Fixed model and data

| Item | Value |
| --- | --- |
| Checkpoint | `runs/20260927T210911Z_kw10-cnn-baseline_6c9d6cdf/checkpoints/best.pt`, SHA-256 `d21f89283a128820e93b00d6e104805b3423fd47f2ed8791ba7775d5bf9d267e` (Milestone 3, epoch 18). The same checkpoint evaluated in Milestones 4 and 5, with the same gates (pinned hash, compatibility, feature fingerprint, clean training tree that is an ancestor of HEAD, agreement with the training-run metadata). |
| Labels | `yes, no, up, down, left, right, on, off, stop, go`, in this order (from the checkpoint) |
| Preprocessing | the checkpoint's window (version 1, 16,000 samples at 16 kHz) and log-mel features (version 1, fingerprint `8368ac49…00ba1`), computed by the existing validated `LogMelSpectrogram` module |
| Probabilities | raw softmax (`T = 1`). The Milestone 4 temperature is not applied. |
| Manifest | `speech_commands_v0.02_kw10.csv`, SHA-256 `f811a20d…ab6131c`, through the validated-manifest gate |
| Real-data split | the official **test** split only (4,074 rows, duplicates retained), used for numerical parity and streaming consistency. Nothing is selected or tuned on it. |
| Clean reference | the Milestone 4 run `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a`, whose `report.json` is pinned by SHA-256 `2f9a0f5bdbbdb89cf1053c604e821ccd139950a73d6fe07ab4e059872d56b237`. Its batch size (256) and thread count (8) are checked as in Milestone 5. |

## 2. Streaming contract `kw10-streaming-v1`

### 2.1 Input

- A stream is mono audio at **16,000 Hz**, the checkpoint window's sample rate. A declared sample rate that differs is rejected: there is no resampling and no channel mixing.
- Audio arrives in **chunks** of any length, including 0, which has no effect. Each chunk is a 1-D NumPy array of one of two types:
  - `int16` PCM, converted exactly as the validated front end does: `float32(x) / 32768`;
  - `float32` samples already on that scale, for example the outputs of the `kw10-robustness-v1` corruptions. They must be finite and are used as they are, not clipped or rescaled.

  Any other dtype or shape is rejected, never converted silently.
- **Timestamps** are integer sample indices counted from the start of the stream (index 0 is the first sample after construction or reset). Seconds are `index / 16000`. Every window reports:
  - `index`;
  - `start_sample` and `end_sample` (`start + 16000`, exclusive);
  - `valid_samples`, the number of real (not padded) samples;
  - `final_partial`;
  - `emitted_at_sample`, the number of samples received when it was emitted, counted in stream time. That is `start_sample + 16000` for a full window, which the push delivering that sample emits however large the chunk, and `n` for the final window. Like every other field, it does not depend on chunking. *(Clarified while implementing, before any run.)*

### 2.2 Windows and hop

- **Window** `W` = 16,000 samples (1.000 s), the checkpoint's `window.num_samples`. **Hop** `H` = 1,600 samples (100 ms), fixed.
- Window `k` covers stream samples `[k·H, k·H + W)`.
- **No startup padding.** Window 0 starts at sample 0 and is emitted only once `W` samples have arrived.

  The training windows (`start_zero_pad`) always begin with real audio and have zeros, if any, only after a short clip. Left zero padding would put digital silence *before* the audio, which the model never saw. Without it, window 0 of a stream is exactly the clip-level input of `kw10-clean-eval-v1`.
- **Emission:** window `k` is emitted as soon as the stream has received `k·H + W` samples, during the push that delivers sample `k·H + W − 1`. Windows are emitted in increasing `k`, never twice, and never revised.
- **End of stream (`flush`):**
  - If some received sample is not in any emitted window, exactly one **final partial window** is emitted. It is the next window of the grid, `K`, holding samples `[K·H, n)` followed by zeros up to `W`, with `valid_samples = n − K·H` and `final_partial = true`.
  - Otherwise nothing more is emitted. Windows made only of already-covered audio and padding are never emitted.
- **Short audio:** a stream with `1 ≤ n < W` samples produces no window until `flush`, which emits window 0: the audio followed by zeros. That is exactly the clip-level window of `kw10-clean-eval-v1`, so every Speech Commands clip (at most 16,000 samples) produces **exactly one window**.
- **Empty stream:** a stream with `n = 0` produces no window and **no decision**. The result is explicitly empty, never a default label.
- After `flush` the stream is finished, and a further `push` or `flush` is an error until `reset`.

### 2.3 State and reset

- The only streaming state is:
  - the buffered samples not yet needed by an emitted window, at most `W` plus one chunk;
  - the window counter;
  - the probability vectors of the last `M` windows (section 2.5).

  The model is stateless: each window is computed from its own 16,000 samples from scratch, with no feature caching and no reuse of overlapping frames.
- **Reset** discards the buffer, the counter, the aggregation history, and the finished flag. The next sample is index 0 again, and the outputs are identical to those of a new instance.

### 2.4 Per-window prediction

- Each window is inferred **on its own, with batch size 1**.

  On the synthetic probes run while writing this protocol, PyTorch CPU outputs for the same window differed by up to about 1e-5 between batch sizes 1 and 64. Batch 1 makes every output independent of how the audio was chunked.
- Logits are float32 in label order. Probabilities are `softmax(logits)` in float64, subtracting the maximum first (the Milestone 4 definition, `T = 1`).
- The window **label** is the argmax of the logits, with ties going to the lowest label index. The **confidence** is the largest probability.

### 2.5 Aggregation (causal; no future windows)

- When window `k` is emitted, its aggregate is `a_k`, the arithmetic mean of the probability vectors of windows `max(0, k − M + 1) … k`, with **`M` = 5**. That is at most 5 windows, whose starts span 400 ms.
- The aggregated label is the argmax of `a_k` (ties go to the lowest index), and the aggregated confidence is its largest value.
- `a_k` uses only windows `0 … k`. It is computed once, when window `k` is emitted, and never changes afterwards. **Aggregation never uses future windows.**
- **Stream (clip) decision:** the aggregated output of the last emitted window after `flush`.
  - For a stream with one window, which includes every clip of at most 1 s, it equals that window's prediction, and therefore the clip-level prediction of `kw10-clean-eval-v1`.
  - For longer streams it reflects the last `M` windows only. The full per-window timeline is always available.
- `M`, `H`, and this rule were fixed here before any streaming output existed, and they are not tuned.

### 2.6 Offline reference and determinism

- The **offline reference** computes the same windows from the complete signal at once, by an independent implementation of section 2.2. It then applies the same per-window inference and aggregation.
- For **any** chunking of the same audio, streaming output must be **bit-identical** to the offline reference: the same windows, timestamps, logits, probabilities, and aggregates. This is tested.
- **Closed set:** every window, including silence, noise, and speech outside the 10 keywords, receives one of the 10 labels. There is no rejection, and no claim of silence or unknown-word handling.

## 3. ONNX export `kw10-onnx-v1`

### 3.1 Graph and toolchain

- **Exported module:** the checkpoint's validated `LogMelSpectrogram` (window, `torch.stft`, mel filterbank, log) followed by `kws_cnn_v1`, in eval mode (batch norm uses its running statistics; dropout is off). The graph maps waveforms to logits. Softmax, argmax, and aggregation stay outside the graph and are shared with the PyTorch path.
- **Exporter:** `torch.onnx.export(..., dynamo=True)`, the `torch.export`-based exporter, with torch 2.14.0+cpu, onnxscript 0.7.2, and onnx 1.23.0. The STFT is exported as the ONNX `STFT` operator. Formal export runs require exactly these versions; a different toolchain is an error with a diagnostic.
- **Opset:** 18, default ONNX domain only. The model must pass `onnx.checker.check_model(full_check=True)`.
- **Tensor contract:**

  | Name | Type | Shape | Meaning |
  | --- | --- | --- | --- |
  | input `waveform` | float32 | `[batch, 16000]`, `batch` dynamic (1–65,536) | one window per row, on the `int16 / 32768` scale |
  | output `logits` | float32 | `[batch, 10]` | label order stored in the model metadata |

- **Determinism:** the export is performed twice in the export run, and the two serialisations must be byte-identical. The model bytes contain no run ID or timestamp. The synthetic probe produced identical bytes, but byte identity across other toolchain versions is not claimed.

### 3.2 Metadata and compatibility checks

- The model's `metadata_props` record:
  - the format (`robust-streaming-ml/onnx`, version 1) and protocol IDs;
  - the checkpoint SHA-256, run ID, and epoch;
  - labels, window, features, feature version, and feature fingerprint;
  - the dataset and manifest SHA-256;
  - the input and output contract and the opset;
  - the export source revision (commit and dirty flag);
  - the export toolchain versions.
- A JSON **sidecar** `model.onnx.json` repeats this metadata. It adds the file SHA-256, the export run ID and time, a graph summary (inputs, outputs, opset, IR version, producer, operator types), and the export environment (Python, platform, package versions).
- **Loading** an exported model fails closed on any of:
  - a file SHA-256 that differs from the sidecar;
  - sidecar metadata that differs from `metadata_props`;
  - an unknown format or version;
  - an opset other than 18, or any non-default domain;
  - a tensor contract that differs from section 3.1;
  - a checker failure;
  - a checkpoint SHA-256, label order, feature fingerprint, window, feature settings, or manifest SHA-256 that differs from what the caller expects.

### 3.3 Runtime

- **ONNX Runtime 1.30.0, `CPUExecutionProvider` only.** The session is created with exactly that provider, and the providers in use are verified. Formal runs require exactly this version: CPython 3.14 on Windows x86-64, as locked in `uv.lock`.
  - The Linux Python 3.10 CI environment locks onnxruntime 1.23.2, because 1.30.0 has no Python 3.10 wheel. Tests run there, but formal results are claimed only for 1.30.0.
- **Session options for parity:**
  - graph optimisation `ORT_ENABLE_ALL`;
  - sequential execution;
  - 1 inter-op thread and 1 intra-op thread.

## 4. Parity and streaming consistency (`kw10-onnx-v1`)

### 4.1 Reference

The reference is the PyTorch checkpoint path, evaluated on CPU in float32 with 8 torch threads, under the streaming contract (one window per call, batch 1).

**Clean-path gate:** before anything else, the PyTorch path is evaluated on the 4,074 test clip windows with batch size 256. It must reproduce the Milestone 4 raw test logits **bit-for-bit**, as in Milestone 5, otherwise the run fails. The reference is therefore the evaluated model.

### 4.2 Parity sets (real test audio)

- **A, clip streams.** Each of the 4,074 test clips is its own stream, pushed as `int16` chunks with seeded pseudo-random sizes and then flushed. Each yields exactly one window, which must be bit-identical to its Milestone 4 clip window, otherwise the run fails.
- **B, concatenation streams.** The test rows, in manifest order, are split into 510 groups of 8 consecutive rows; the last group has 2. Each group's raw clips are concatenated without gaps into one stream, so windows straddle clip boundaries at arbitrary offsets.
  - Each stream is pushed in seeded pseudo-random chunk sizes and flushed. Every window it emits must be bit-identical to the offline reference windows, otherwise the run fails.
  - Parity is compared on every window. The windows have no single true label, so no accuracy is computed for B.
- **Chunk sizes:** log-uniform integers in `[1, 32000]` from `numpy.random.default_rng(derive_seed(2718, "kw10-onnx-v1", "chunks", <stream id>))`. The stream ID is the clip's `relative_path` for A and `stream-<index>` for B. Chunking cannot change outputs (section 2.6); it only exercises the engine on real audio.

### 4.3 Tolerances (fixed here, before any real parity run)

| Quantity, per window | Tolerance |
| --- | --- |
| Logits, every class | `\|onnx − torch\| ≤ 1e-2` |
| Raw probabilities, every class | `\|onnx − torch\| ≤ 1e-3` |
| Aggregated probabilities, every class (set B, and A) | `\|onnx − torch\| ≤ 1e-3` |
| Top-1 label | must agree, unless the reference top-2 logit margin is `≤ 2e-2` (twice the logit tolerance). Only then can a flip be consistent with the logit tolerance. Such flips are counted and reported. |
| Aggregated top-1 label | must agree, unless the reference aggregated top-2 margin is `≤ 2e-3`; such flips are counted |

**Basis.** These tolerances come from synthetic probes run while writing this protocol: random weights, and the pinned checkpoint on synthetic signals only, with no dataset audio.
- The largest differences were `|Δlogit|` 1.7e-3 and `|Δprob|` 2.7e-4, on pure tones.
- On speech-like, band-limited, and all 15 `kw10-robustness-v1`-corrupted synthetic signals, they were at most 1.3e-4 and 1.2e-5.
- The front end's `log(mel + 1e-6)` amplifies FFT rounding differences in almost-empty spectral bands.

**Any violation fails parity.** The parity artifacts and a report listing every violation are still written, the run's status is `failed`, and the command exits non-zero. Later milestones (benchmark and demo) refuse an export whose parity did not pass.

### 4.4 Reported quantities

- **For each parity set:**
  - the number of windows;
  - the maximum and mean `|Δlogit|` and `|Δprob|`;
  - the maximum aggregated `|Δprob|`;
  - top-1 and aggregated top-1 disagreements, split into near-tie flips and violations;
  - pass or fail.
- **Streaming consistency (set A):** the DESIGN metric "window prediction consistency". It counts how often the **stream decision** of the deployment path (ONNX Runtime, streaming, batch 1) equals the Milestone 4 clip-level prediction, over all 4,074 clips. The same count is reported for the PyTorch streaming path, which uses batch 1 while Milestone 4 used 256. The accuracy of both streaming paths is reported next to the Milestone 4 accuracy.

  Agreement is **measured and reported, not assumed**. The window identity is a gate; label agreement is not.

## 5. Outputs

The export run directory `runs/<UTC time>_kw10-cnn-baseline-onnx_<suffix>/` is git-ignored, like every model and ONNX file. It contains:
- `config.yaml` and `metadata.json` from `start_run`;
- `onnx/model.onnx` and `onnx/model.onnx.json`;
- `parity/clips.csv`: per test clip, the true label, the Milestone 4 prediction, the PyTorch and ONNX logits (exact float32), and both stream decisions;
- `parity/stream_windows.csv`: per set-B window, the stream, the timestamps, and the PyTorch and ONNX logits;
- `parity/streams.csv`: per set-B stream, its member clips, its length, and its window and chunk counts;
- `report.json`: the protocols, toolchain and runtime versions, session options, checkpoint, baseline, the clean-path gate, the parity statistics and violations, streaming consistency, and the SHA-256 of every artifact;
- `report.md` and `summary.json`.

Every parity statistic can be regenerated from the CSV files. Files are written atomically.

## 6. Known limitations

- **Parity scope:** parity is measured on clean test audio only (sets A and B). Corrupted audio is covered only by the synthetic probes and tests, not by a real-data parity run.
- **Dataset shape:** Speech Commands clips are isolated utterances of at most 1 s. Real-data streaming therefore has one window per clip (set A), or artificial concatenations (set B) with no per-window ground truth. There is no evaluation on natural continuous audio.
- **Closed set:** no silence or unknown class, so streams always yield a keyword label. The stream decision of a long stream reflects only its last 5 windows.
- **One of everything:** one model, one seed, one runtime version, one platform. Parity numbers are specific to this toolchain.
- **No latency claims** in this milestone.
