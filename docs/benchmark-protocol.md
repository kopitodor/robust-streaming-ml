# CPU latency benchmark protocol `kw10-latency-v1`

Milestone 7. This protocol was written and committed **before** the benchmark code and before any latency was measured. Results will go in [RESULTS.md](../RESULTS.md).

It measures per-window CPU latency of the deployment-oriented streaming path on one local machine. It is a **descriptive measurement of this machine under the stated conditions**. It is not a production service-level agreement, a real-time guarantee, or a claim about other hardware.

## 1. What is measured

| Item | Value |
| --- | --- |
| Model | the Milestone 6 ONNX export `onnx/model.onnx`, SHA-256 `e1db0406461abb4649cd6ab36a76b2369067402a19ef2d80823825a9b0ea9e8b`, from run `20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b`. Its `report.json` is pinned by SHA-256 `7fddcac2a6e6bbc08b14620e35c1bc1550598e575ecb1f92b2aab9cc2da4fa60` and must record passed parity. |
| Checkpoint | `d21f89283a128820e93b00d6e104805b3423fd47f2ed8791ba7775d5bf9d267e`: the source of the export, and the PyTorch reference path |
| Runtime | ONNX Runtime **1.30.0**, `CPUExecutionProvider` only, with `ORT_ENABLE_ALL` graph optimisation, sequential execution, and 1 inter-op thread. The version and provider are checked; any other is an error. |
| Streaming | `kw10-streaming-v1`: window 16,000 samples (1 s), hop 1,600 samples (100 ms), **batch size 1** (one window per inference), raw probabilities, causal mean of the last 5 windows |
| Input audio | real audio: official test clips concatenated without gaps in manifest order into one stream, taking only as many clips as the benchmark needs. It is read through the validated-manifest gate and re-verified against the manifest. |

### Measured paths

Each path is timed in a separate pass, and each is labelled in every result.

1. **`onnx_streaming_step` (primary; deployment-oriented).** One call to `StreamingClassifier.push(chunk)` with the next 1,600 `int16` samples, which completes exactly one window.
   - **Included:**
     - chunk validation and the `int16 → float32` conversion;
     - buffer append and trim;
     - window extraction (a copy);
     - **ONNX Runtime inference, including feature extraction**: the log-mel front end (STFT, mel, log) is inside the ONNX graph;
     - float64 softmax, argmax, and the causal aggregate update.
   - **Excluded:** reading and decoding audio files, corruption, session creation and model loading, `flush`, and output formatting.
2. **`onnx_session_run` (secondary breakdown).** `InferenceSession.run` alone on a pre-extracted `[1, 16000]` float32 window: feature extraction and model, without the streaming bookkeeping. The windows are the same ones the streaming path sees.
3. **`pytorch_streaming_step` (secondary reference).** The same streaming step as path 1, with the PyTorch eager backend (the validated front end and CNN, `torch.no_grad`, `torch.use_deterministic_algorithms(True)` as set by `start_run`).

   It is reported only as the non-exported reference implementation on the same machine. It is **not** a claim about optimised PyTorch deployment.

### Thread configurations

Every path is measured with **1** and **4** intra-op threads:
- ONNX Runtime `intra_op_num_threads`;
- `torch.set_num_threads` for the PyTorch path; the torch inter-op pool is left at its default and recorded.

The 1-thread configuration is primary.

## 2. Procedure

- **Warm-up:** 500 steps (windows) per measurement block, not timed.
- **Timed iterations:** 5,000 steps per block. Each step is timed individually.
- **Repetitions:** 3 blocks per path and thread configuration, in a fixed interleaved order:

  ```
  for repetition in 1..3:
    for threads in (1, 4):
      for path in (onnx_streaming_step, onnx_session_run, pytorch_streaming_step):
        block
  ```

  Each block creates a fresh session or streaming classifier, which is not timed. It pre-fills 14,400 samples, not timed, so that every following push of 1,600 samples completes one window. Every block uses the same stream positions, so every block sees the same 5,500 windows.
- **Timer:** `time.perf_counter_ns()` immediately before and after the measured call. The clock's implementation and resolution from `time.get_clock_info("perf_counter")` are recorded.
- **Sanity checks, before timing, not timed:**
  - For the first 20 benchmark windows, the ONNX streaming logits must be bit-identical to ONNX Runtime on the offline reference windows.
  - They must agree with the PyTorch path within the `kw10-onnx-v1` parity tolerances.
  - During timing, every streaming step must emit exactly one window.

  Any failure stops the run.
- **Process policy:**
  - a single Python process at normal priority;
  - no CPU affinity pinning, and no change to the power plan (neither is recorded as controlled);
  - the Python garbage collector is left enabled, as in normal use;
  - other applications on the machine are not closed or controlled.

## 3. Statistics

- **Percentiles, nearest rank:** for `N` sorted durations `x(1) ≤ … ≤ x(N)`, `p_q = x(⌈q·N⌉)`.
  - p50 is `x(⌈0.50·N⌉)` and p95 is `x(⌈0.95·N⌉)`. No interpolation.
- **Primary values:** p50 and p95 over the **pooled** timed steps of the 3 repetitions of a path and thread configuration (`N` = 15,000).
- **Also reported:** per-repetition p50 and p95 (to show run-to-run spread), mean, minimum, maximum, and p99.
- All values are in milliseconds, computed from integer nanoseconds.
- Derived "real-time factor" figures are **not** reported, and no real-time capability is claimed.

## 4. Recorded environment

`report.json` records:
- **OS:** `platform.platform()`, the OS release and version, and the machine architecture.
- **CPU:**
  - the brand string (Windows registry `ProcessorNameString`; Linux `/proc/cpuinfo` `model name`);
  - the nominal MHz where the OS reports it;
  - logical cores (`os.cpu_count()`);
  - physical cores when available (Windows `GetLogicalProcessorInformationEx`; Linux unique core IDs), otherwise `null`.
- **Versions:** Python; ONNX Runtime, and its available providers; torch, with its intra-op and inter-op thread counts; numpy.
- **Timer:** the clock information.
- **Session options:** as used, per configuration.
- **Model identity:** the ONNX model and checkpoint SHA-256, the export run, and the source revision from `start_run`.

## 5. Outputs

The run directory `runs/<UTC time>_kw10-onnx-latency_<suffix>/` is git-ignored. It contains:
- `config.yaml` and `metadata.json` from `start_run`, which requires a clean source tree;
- `timings.csv`: **every timed step**, raw. Columns: `path`, `threads`, `repetition`, `step`, `window_index`, and `ns`. Warm-up steps are not stored.
- `report.json`: the protocol, environment, sanity checks, and per-configuration statistics (pooled and per repetition), plus the SHA-256 of `timings.csv`;
- `report.md` and `summary.json`.

Every statistic can be regenerated from `timings.csv`.

## 6. Limitations

- **One machine:** a laptop, with the power plan, thermal state, turbo behaviour, and background load **not controlled**. Numbers can differ between runs on the same machine, which is why the per-repetition spread is shown, and on any other machine.
- **Single stream:** batch size 1. The numbers include Python-level overhead of the streaming wrapper and the ONNX Runtime Python API.
- **Scope:** no audio input/output, decoding, or end-to-end application latency is included. Model loading and session creation are excluded.
- **No claims:** this is not a production benchmark, a service-level agreement, or a real-time claim. The PyTorch path is an unoptimised eager reference, not a tuned deployment.
