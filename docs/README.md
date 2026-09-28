# Documentation

Supporting protocol notes, dataset license records, demo instructions, and reproducibility reports will live here.

- [demo.md](demo.md): the local, offline demo. It covers requirements, launch commands, what the terminal and HTML page show, how the output is produced, and limitations (closed set, no silence or unknown-word rejection, not a result, not production-ready).
- [benchmark-protocol.md](benchmark-protocol.md): CPU latency protocol `kw10-latency-v1`. It covers the pinned ONNX export and runtime; the measured paths (the ONNX streaming step including feature extraction, `session.run` alone, and a PyTorch eager reference); thread settings; warm-up, iterations, and repetitions; the timer; nearest-rank p50 and p95; recorded hardware; and limitations. Written before any measurement.
- [streaming-onnx-protocol.md](streaming-onnx-protocol.md): streaming contract `kw10-streaming-v1` (input, 1 s window, 100 ms hop, timestamps, no startup padding, one final partial window, short audio, reset, batch-1 per-window prediction, causal 5-window aggregation) and export/parity protocol `kw10-onnx-v1` (graph, opset 18, tensor contract, metadata checks, ONNX Runtime CPU, predeclared tolerances). Written before the implementation.
- [robustness-protocol.md](robustness-protocol.md): protocol `kw10-robustness-v1`, with its fixed inputs and gates, the 5 corruptions × 3 severities with parameters, seeds, the codec backend and licensing, output validation, degradation, the duplicate policy, and limitations.
- [evaluation-protocol.md](evaluation-protocol.md): protocol `kw10-clean-eval-v1`. It defines the gates, test isolation, metric definitions (zero-division policy, 15-bin ECE), temperature scaling fitted on validation only, the duplicate policy, and the outputs.
- [baseline.md](baseline.md): the Milestone 3 baseline: window and log-mel features, CNN shapes and parameter count, training procedure, determinism, checkpoint format, and run layout.
- [data-contract.md](data-contract.md): Speech Commands v2 10-keyword data contract (source, license verification fields, splits, validation rules, manifest schema). Planning; not yet implemented.
- [datasets/speech_commands_v0.02.md](datasets/speech_commands_v0.02.md): verification record for the downloaded archive (checksums, license, attribution, structural cross-checks).
- [decisions/](decisions/): architecture/dataset decision records.
  - [0001-dataset-speech-commands-v2.md](decisions/0001-dataset-speech-commands-v2.md): dataset choice and scope.
