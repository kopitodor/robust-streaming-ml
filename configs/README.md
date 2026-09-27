# Configurations

Versioned training, evaluation, corruption, streaming, and benchmark configurations will live here.

- `kw10_cnn_baseline.yaml`: the Milestone 3 baseline, validated by `TrainingConfig` (`schema_version` 1). It has `data` (the pinned manifest, validation report, audio root, and label order), `window` (version 1), `features` (log-mel, version 1), `model` (`kws_cnn_v1`), `training`, and `checkpoint` sections. The labels are quoted because YAML 1.1 reads bare `yes`/`no`/`on`/`off` as booleans. See `docs/baseline.md`.
- `example.yaml` — minimal configuration exercising the Milestone 1 schema (`name`, `seed`, `deterministic`, `run.output_root`). It defines no dataset or model and is not an experiment.

All files are validated with `robust_streaming_ml.config.load_config` (training configs with `schema=TrainingConfig`).
