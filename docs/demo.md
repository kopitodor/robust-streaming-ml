# Local demo

A local, offline demo of the streaming keyword classifier. It runs the pinned Milestone 6 ONNX export, whose export report records passed PyTorch/ONNX parity (`kw10-onnx-v1`), under the fixed streaming contract `kw10-streaming-v1`.

It is a way to *look at* the model's behaviour on a file of your choice. It is **not** an evaluation, a benchmark, or a production system.

## What you need

- **The environment:** `uv sync`, which installs ONNX Runtime and the pinned `soundfile` codec backend.
- **The Milestone 6 export run**, a local, git-ignored artifact: `runs/20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b/`, pinned in [`configs/kw10_demo.yaml`](../configs/kw10_demo.yaml) by its report SHA-256. Without it the demo refuses to start.

  To reproduce it:
  1. acquire the dataset and build the manifest;
  2. train;
  3. run the evaluation;
  4. run `scripts/export_onnx.py`;
  5. pin the new export report in the demo config.

  See the README.
- **A 16 kHz, mono, 16-bit PCM WAV file**, up to 60 s long (`demo.max_seconds`). Any Speech Commands clip qualifies. Other formats are refused with a message; the demo does not resample or mix channels.

## Launch

```bash
uv run python scripts/demo.py --audio path/to/clip.wav
uv run python scripts/demo.py --audio path/to/clip.wav --corruption resample --severity via_2000hz --open
uv run python scripts/demo.py --list-corruptions
```

| Option | Meaning |
| --- | --- |
| `--audio` | the local WAV file (required) |
| `--corruption`, `--severity` | one `kw10-robustness-v1` corruption and severity (both or neither). It is applied to the whole file as one clip, seeded from the audio content, so reruns give identical output. |
| `--output` | the HTML path (default: `artifacts/demo/<file>[_<corruption>_<severity>]_<UTC time>.html`); the JSON record goes next to it |
| `--open` | opens the local page in your default browser |
| `--config` | defaults to `configs/kw10_demo.yaml` |

## What it shows

**In the terminal:** for every window, its index, time span, predicted label and confidence, the causal aggregate's label and confidence, and ONNX Runtime inference time. Then the stream decision, for the clean input and, if requested, for the corrupted input.

**On the HTML page:**
- **Stream decision cards** for the clean and corrupted input, and how many window labels the corruption changed.
- **Players** for the clean audio and the corrupted audio, embedded as `data:` URIs so the page stays offline.
- **A timeline per stream:**
  - one bar per window at its start time: its height is the top-1 confidence, and its colour is the label;
  - a line for the aggregate confidence;
  - a strip for the aggregate label.

  Hover over a bar for its details.
- **A per-window table** (clean and corrupted side by side), which marks the final zero-padded window.
- **Timing measured during this run:** the median and maximum ONNX Runtime inference time per window, and the stream's wall time. The protocol benchmark is `kw10-latency-v1` in [RESULTS.md](../RESULTS.md).
- **Model and provenance:**
  - the ONNX model SHA-256, the export run, and the pinned export report;
  - whether parity passed;
  - the checkpoint SHA-256, run, and epoch;
  - the label order, opset, and feature fingerprint;
  - the runtime and provider;
  - the demo's code revision.
- **Training data:** Speech Commands v0.02 (10-keyword subset), the manifest SHA-256, the CC BY 4.0 licence, the attribution, and the citation.
- **The streaming contract and the demo configuration.**
- **Limitations.**

## How the output is produced

1. The pinned export is loaded through the same fail-closed checks as the benchmark:
   - the report SHA-256;
   - passed parity;
   - the model file hash and sidecar;
   - the checkpoint, labels, feature fingerprint, and manifest.
2. The file is pushed through `StreamingClassifier` in 1,600-sample (100 ms) chunks, then flushed:
   - 1 s windows every 100 ms, with no startup padding;
   - one zero-padded final window if audio is left over;
   - every window inferred on its own on ONNX Runtime (`CPUExecutionProvider`, 1 intra-op thread by default);
   - raw softmax, and the aggregate as the mean of the last 5 windows' probabilities;
   - the stream decision is the aggregate of the last window.

   A file of at most 1 s therefore gives exactly one window, and its decision is the clip-level prediction.
3. With a corruption, the corrupted samples (float32, on the `int16 / 32768` scale) go through a fresh classifier in the same way.

## Limitations

- **Not a result:** demo output describes one file on one machine. Nothing it shows is a measured or reported result.
- **Closed set:** every window is assigned one of `yes, no, up, down, left, right, on, off, stop, go`, including silence, noise, music, and other words. There is **no silence or unknown-word rejection**.
- **Long files:** the stream decision reflects only the last 5 windows. With several words or trailing silence, the per-window timeline is the informative view.
- **Scope:** the model was trained and evaluated on isolated 1 s Speech Commands clips, with one model and one seed. Behaviour on continuous or real-world audio, other microphones, accents, or noise was not evaluated.
- **Corruptions** are the synthetic `kw10-robustness-v1` transforms, applied to the whole file. They are not recordings of real conditions.
- **Timings** come from this run on this machine, with Python overhead. They are not a benchmark or a real-time guarantee.
- **No production readiness** is claimed.
