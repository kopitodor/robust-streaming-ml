# Design

## Problem statement

Build a bounded streaming audio-classification system using one supported public speech or audio dataset. The project will show how a small, understandable classifier behaves when audio is processed in realistic streaming windows and under controlled signal corruption. It is a portfolio project, not a production service or a state-of-the-art benchmark.

## User-facing demo definition

Given an audio clip (or microphone-like sequential windows), a local demo will display the predicted class, confidence, recent-window aggregation behavior, and inference timing. A selected controlled corruption can be applied so the demo can contrast clean and corrupted predictions. The demo will identify the exact model, dataset version, split manifest, and corruption configuration used; it will not imply live production readiness.

## Architecture proposal

1. Acquire a public, license-reviewed dataset and produce an immutable local manifest.
2. Validate files, labels, sample rates, duration constraints, duplicates, and train/validation/test split membership before training.
3. Convert fixed-duration streaming windows to log-mel spectrogram features with versioned preprocessing settings.
4. Train a modest PyTorch CNN baseline; retain configuration, random seeds, dependency lock information, checkpoints, and split manifest.
5. Evaluate clean data, calibration, and deterministic corruption scenarios.
6. Export the inference path to ONNX and run CPU streaming-window benchmarks with a documented runtime and hardware description.

## Data and licensing assumptions

The dataset is not selected or downloaded in this foundation. It must be a publicly accessible speech or audio dataset whose license permits the planned local portfolio use and whose redistribution constraints are respected. Dataset version, source URL, license, attribution requirements, class definition, and acquisition checksum/manifest must be recorded before use. No unsupported, private, ambiguously licensed, or fabricated data is in scope.

## Exact metrics to collect

All metrics are computed separately for validation and held-out test splits where applicable:

| Area | Metric | Definition |
| --- | --- | --- |
| Classification | Accuracy | Correct predictions divided by evaluated examples |
| Classification | Macro F1 | Unweighted mean of per-class F1 scores |
| Errors | Confusion matrix | Counts by true class and predicted class |
| Calibration | Expected calibration error (ECE) | Binned confidence-to-accuracy gap, with bin count documented |
| Calibration | Negative log-likelihood (NLL) | Mean negative log probability of the true class |
| Robustness | Per-corruption accuracy and macro F1 | Metrics for each corruption type and severity |
| Robustness | Degradation | Clean metric minus corrupted metric at the same split and protocol |
| Streaming | Window prediction consistency | Agreement between specified window aggregation and clip-level protocol |
| Performance | CPU latency p50 and p95 | Per-window end-to-end inference latency with warm-up, batch size, runtime, threads, and hardware documented |

## Acceptance criteria

- A single supported dataset is acquired through a documented, reproducible procedure and passes validation.
- Split manifests are deterministic, disjoint, class-stratified when applicable, and checked for leakage and duplicates.
- The baseline trains from versioned configuration and a documented seed, producing reloadable artifacts.
- Clean validation/test metrics, a confusion matrix, and calibration outputs are generated from a repeatable evaluation command.
- Noise, reverberation, resampling, clipping, and codec-artifact evaluations run under explicit, deterministic configurations and report degradation against clean evaluation.
- Streaming windows yield valid predictions; the exported ONNX model matches the reference protocol within predeclared numerical tolerances.
- CPU p50/p95 latency is measured under a reproducible documented benchmark protocol.
- Documentation distinguishes measured results from plans and includes no unsupported claims.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| License or access ambiguity | Review and record source, version, license, attribution, and intended use before acquisition. |
| Leakage or duplicate audio | Create manifests first; validate split disjointness, hashes, and grouping keys before training. |
| Irreproducible experiments | Version configs and manifests, set/report seeds, and capture package/runtime metadata. |
| Corruptions that are unrealistic or inconsistent | Use documented transforms, fixed severity definitions, deterministic seeds, and retain generated manifests. |
| Streaming/offline mismatch | Specify window, hop, padding, and aggregation semantics; test them directly. |
| Export/runtime divergence | Compare PyTorch and ONNX outputs on a fixed test fixture before benchmarking. |
| Misleading portfolio presentation | Publish only measured, traceable results with protocol and limitation notes. |

## Non-goals

- No fabricated claims or metrics.
- No state-of-the-art chase.
- No production, reliability, or real-time claims.
- No unsupported datasets.

## Valid portfolio result

A valid result is a reproducible, traceable run on a documented supported dataset: it includes data/license provenance, split manifest, configuration and seed, code revision/dependency context, exact evaluation protocol, raw or regenerable outputs, and honest limitations. It may report modest performance. It is invalid to report planned numbers, selectively omit corruption failures, compare incompatible protocols, or infer production suitability from this project.
