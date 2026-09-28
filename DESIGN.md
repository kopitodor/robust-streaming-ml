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

**Selected, acquired, and verified (2026-09-27):** Google Speech Commands v2 (`v0.02`), limited to the ten labels `yes, no, up, down, left, right, on, off, stop, go`, with the official validation/test lists. `_silence_` and `_unknown_` are out of scope, so it is a closed-set classifier. The archive checksum, the license (CC BY 4.0, with the project owner's sign-off), and the attribution were recorded from the actual download. See [docs/decisions/0001-dataset-speech-commands-v2.md](docs/decisions/0001-dataset-speech-commands-v2.md), the [data contract](docs/data-contract.md), and the [verification record](docs/datasets/speech_commands_v0.02.md).

The general requirements still apply. The dataset must be a publicly accessible speech or audio dataset whose license permits the planned local portfolio use and whose redistribution constraints are respected. Dataset version, source URL, license, attribution requirements, class definition, and acquisition checksum/manifest must be recorded before use. No unsupported, private, ambiguously licensed, or fabricated data is in scope.

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

## Status against the acceptance criteria (2026-09-28)

Evidence is in [RESULTS.md](RESULTS.md). Every run below used a clean, committed source tree, and each protocol was committed before its first run.

| Criterion | Status | Evidence and caveats |
| --- | --- | --- |
| A single supported dataset is acquired through a documented, reproducible procedure and passes validation | Met | `scripts/acquire_speech_commands.py` (pinned archive SHA-256) and `scripts/build_speech_commands_manifest.py`: validation passed with 0 errors, and the manifest SHA-256 `f811a20d…` is pinned. A second build was byte-identical. |
| Split manifests are deterministic, disjoint, class-stratified when applicable, and checked for leakage and duplicates | Met, with ADR-001 | The official speaker-based lists are used, so class stratification is not applicable (ADR-001). Splits are disjoint by path, speaker, and content hash. The 1,477 within-split duplicate warnings (640 groups, each from a single speaker) are reported and handled by a declared evaluation policy. Near-duplicates are not detected. |
| The baseline trains from versioned configuration and a documented seed, producing reloadable artifacts | Met | Run `20260927T210911Z_kw10-cnn-baseline_6c9d6cdf` (seed 1234, deterministic). The versioned checkpoint reloads with `weights_only=True`. |
| Clean validation/test metrics, a confusion matrix, and calibration outputs come from a repeatable evaluation command | Met | Run `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` (`kw10-clean-eval-v1`); a reproducibility re-run gave identical outputs. |
| Noise, reverberation, resampling, clipping, and codec-artifact evaluations run under explicit, deterministic configurations and report degradation against clean evaluation | Met | Run `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` (`kw10-robustness-v1`): 15 conditions, per-clip seeds, degradation against the Milestone 4 raw test metrics. The corruptions are synthetic. |
| Streaming windows yield valid predictions; the exported ONNX model matches the reference protocol within predeclared numerical tolerances | Met | Run `20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b` (`kw10-streaming-v1`, `kw10-onnx-v1`). Parity passed on 39,703 windows (max \|Δlogit\| 1.8e-4 against a 1e-2 tolerance), and streaming decisions equal the clip-level predictions on 4,074/4,074 clips. Streaming was evaluated on isolated clips and artificial concatenations only. |
| CPU p50/p95 latency is measured under a reproducible documented benchmark protocol | Met | Run `20260928T004340Z_kw10-onnx-latency_1df4024f` (`kw10-latency-v1`), with raw timings retained. It covers one uncontrolled local machine and one run, with no production or real-time claim. The report's `torch_threads` field was mis-captured; this is disclosed, and the timings are unaffected. |
| Documentation distinguishes measured results from plans and includes no unsupported claims | Met, to the best of the recorded review | RESULTS.md lists only executed commands, run IDs, and verified artifacts, including failures and disclosed issues. The CI workflow has not run on GitHub; only a local CPython 3.10 rehearsal is recorded. |

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
- No `_silence_` or `_unknown_` classes, and no out-of-vocabulary rejection claims, in the first baseline.

## Valid portfolio result

A valid result is a reproducible, traceable run on a documented supported dataset: it includes data/license provenance, split manifest, configuration and seed, code revision/dependency context, exact evaluation protocol, raw or regenerable outputs, and honest limitations. It may report modest performance. It is invalid to report planned numbers, selectively omit corruption failures, compare incompatible protocols, or infer production suitability from this project.

## Architecture decision records

### ADR-001: Official speaker-based splits instead of class-stratified splits (2026-09-27)

**Conflict.** The acceptance criteria ask for split manifests that are "class-stratified when applicable". Speech Commands v2 comes with official validation/test lists, assigned per speaker by a hash of the speaker ID rather than stratified by class.

**Decision.** Use the official lists unchanged, and derive train as all remaining files among the ten labels. Class stratification is therefore not applicable. Instead, the manifest reports per-split, per-label counts. Validation fails if any label is missing from any split, if a speaker appears in more than one split, or if audio content (SHA-256) crosses splits.

**Reason.** Speaker-disjoint splits prevent a more serious leakage than class imbalance does, and the official lists keep results comparable with published work. Details: [docs/data-contract.md](docs/data-contract.md).

