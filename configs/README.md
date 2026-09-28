# Configurations

Versioned training, evaluation, corruption, streaming, and benchmark configurations will live here.

- `kw10_cnn_baseline.yaml`: the Milestone 3 baseline, validated by `TrainingConfig` (`schema_version` 1). It has `data` (the pinned manifest, validation report, audio root, and label order), `window` (version 1), `features` (log-mel, version 1), `model` (`kws_cnn_v1`), `training`, and `checkpoint` sections. The labels are quoted because YAML 1.1 reads bare `yes`/`no`/`on`/`off` as booleans. See `docs/baseline.md`.
- `kw10_cnn_baseline_eval.yaml`: formal evaluation (`EvaluationConfig`, `schema_version` 1). It names the validated manifest, the checkpoint pinned by SHA-256, and runtime settings. Protocol definitions are fixed in code; see `docs/evaluation-protocol.md`.
- `kw10_cnn_baseline_robustness.yaml`: the robustness suite (`RobustnessConfig`, `schema_version` 1). It names the validated manifest, the pinned checkpoint, and the pinned Milestone 4 baseline run (report SHA-256). Corruption definitions are fixed in code; see `docs/robustness-protocol.md`.
- `kw10_cnn_baseline_onnx.yaml`: ONNX export and parity (`ExportConfig`, `schema_version` 1). It names the validated manifest, the pinned checkpoint, and the pinned Milestone 4 baseline run. The streaming contract, opset, runtime, parity sets, and tolerances are fixed in code; see `docs/streaming-onnx-protocol.md`.
- `kw10_onnx_benchmark.yaml`: the CPU latency benchmark (`BenchmarkConfig`, `schema_version` 1). It names the validated manifest (for the benchmark audio), the pinned checkpoint, and the pinned Milestone 6 export run (report SHA-256). Paths, threads, warm-up, iterations, and statistics are fixed in code; see `docs/benchmark-protocol.md`.
- `example.yaml` — minimal configuration exercising the Milestone 1 schema (`name`, `seed`, `deterministic`, `run.output_root`). It defines no dataset or model and is not an experiment.

All files are validated with `robust_streaming_ml.config.load_config` (training configs with `schema=TrainingConfig`).
