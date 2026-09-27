# Evaluation protocol `kw10-clean-eval-v1`

Milestone 4. This protocol was fixed and committed **before** any validation or test evaluation was produced. Results are recorded in [RESULTS.md](../RESULTS.md). The implementation is `robust_streaming_ml.evaluation`, run by `scripts/evaluate_baseline.py` with [`configs/kw10_cnn_baseline_eval.yaml`](../configs/kw10_cnn_baseline_eval.yaml).

## 1. What is evaluated

- **Model:** the checkpoint selected in Milestone 3 by the lowest validation loss. That is `runs/20260927T210911Z_kw10-cnn-baseline_6c9d6cdf/checkpoints/best.pt` (epoch 18), with SHA-256 `d21f89283a128820e93b00d6e104805b3423fd47f2ed8791ba7775d5bf9d267e`, pinned in the evaluation config. No other checkpoint (for example `last.pt`) is evaluated on the test split. Neither the model, the preprocessing, nor the calibration method changes in response to test results.
- **Data:** the validated manifest `speech_commands_v0.02_kw10.csv` (SHA-256 `f811a20d…ab6131c`), using the official **validation** split (3,703 clips) and **test** split (4,074 clips) unchanged. The train split is never loaded.
- **Scope:** clean audio only, closed set over 10 labels (ADR 0001). No silence or unknown class.

## 2. Gates, checked before anything is written

1. The checkpoint's file SHA-256 equals the pinned value and its JSON sidecar; it loads with `weights_only=True`; its format version, architecture, and state dict are compatible.
2. The checkpoint's manifest SHA-256 and labels equal the evaluation config's. Its window and feature settings are valid, and its feature fingerprint equals the one this code computes.
3. **Checkpoint provenance:** the checkpoint was trained from a clean tree (`source.dirty` is false). Its commit exists in this repository and is an ancestor of, or equal to, the evaluation commit. The training run's `metadata.json` (two directories above the checkpoint) agrees with the checkpoint on `run_id`, `source`, and `config_sha256`.
4. **Manifest gate:** the same gate as training. The manifest SHA-256 must equal the pinned value; the validation report must be `passed`; the sidecar and report hashes must be consistent; the rows must pass `check_rows`. The checkpoint's window must fit the longest clip.
5. **Evaluation provenance:** the working tree is clean, unless `--allow-dirty-source` is given, and `start_run` records the commit and seeding.

Only the validation and test splits are read. Every file is re-verified against its manifest row (SHA-256 and strict WAV format) before use.

## 3. Order of operations (test isolation)

1. Run inference on validation and write `predictions/validation.csv`.
2. Fit temperature scaling on the validation predictions (read back from that file) and write `calibration.json`.
3. Only then run inference on test and write `predictions/test.csv`.
4. Compute every metric from the prediction files and write `report.json`, `report.md`, and `summary.json`.

The calibration fit refuses predictions from any split other than validation. Test predictions do not exist when the temperature is fitted.

## 4. Inference and probabilities

- The model is in eval mode (batch norm uses its running statistics; dropout is off), on CPU, in float32, batch size 256, with deterministic algorithms on. Front-end settings come from the checkpoint.
- Logits are stored exactly: the float32 values are written with a repr that round-trips.
- **Predicted label:** the argmax of the logits, with ties going to the lowest label index.
- **Probabilities:** `softmax(logits / T)` in float64, subtracting the maximum first. Raw probabilities use `T = 1`.
- All aggregates are recomputed from the stored logits in the prediction files, so every number in the report can be regenerated from those files.

## 5. Metrics

Labels are always in manifest order: `yes, no, up, down, left, right, on, off, stop, go`.

| Metric | Definition |
| --- | --- |
| Accuracy | correct predictions / evaluated examples |
| Confusion matrix | `C[i][j]` = number of examples with true label `i` and predicted label `j` (rows are the truth, columns the prediction, in the fixed label order) |
| Per-class precision, recall, F1 | `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)`. **Zero-division policy:** any 0/0 is 0, so a class with no true positives has F1 0, including a class absent from both truth and predictions. |
| Macro F1 | unweighted mean of the 10 per-class F1 values, over every label, including labels with zero support |
| NLL | mean over examples of `−log p(true label)`, natural log, computed from float64 log-softmax |
| ECE | top-label expected calibration error: confidence = max probability. **15 equal-width bins** over [0, 1]; bin `k = min(floor(confidence × 15), 14)`, so the bins are `[k/15, (k+1)/15)` and the last one includes 1.0. `ECE = Σ_b (n_b / N) · |accuracy_b − mean confidence_b|` over non-empty bins. The report lists each bin's count, accuracy, and mean confidence. |

Each metric has hand-computed test fixtures (`tests/test_metrics.py`).

## 6. Calibration

- **Method:** temperature scaling, one scalar `T > 0`, applied as `softmax(logits / T)`. It is the only calibration method considered, and it is always fitted and reported.
- **Fitting data:** all 3,703 official validation predictions, including duplicates.
- **Objective and search:** minimise the validation NLL over `β = 1/T`, which is convex in `β`. The search is a deterministic golden-section search over `β ∈ [0.05, 20]` with 200 iterations, and the final `β` is the midpoint of the last interval. An optimum at a bound is recorded as `at_bound: true`.
- **Persistence:** `calibration.json` records the method, `T`, `β`, the search settings, the NLL before and after on validation, the fitting inputs (split, row count, prediction file SHA-256, manifest SHA-256), and compatibility metadata (checkpoint SHA-256, run ID, epoch, labels, feature fingerprint). `load_calibration` rejects a file whose checkpoint or labels do not match.
- **Reporting:** raw results come first, then calibrated. Calibrated validation metrics are in-sample, because `T` was fitted on validation, and are labelled as such. Calibrated test metrics are the held-out measurement of the calibration's effect. Temperature scaling cannot change the argmax, so accuracy, macro F1, and the confusion matrix are the same raw and calibrated. Only NLL and ECE can change.

## 7. Duplicate audio

The manifest keeps within-split duplicates (data contract §6). Each group is one speaker and one label, and no group crosses splits:

| Split | Duplicate groups | Files in groups | Redundant copies |
| --- | ---: | ---: | ---: |
| validation | 11 | 22 | 11 |
| test | 119 | 240 | 121 |

- **Official result:** every row of each split, with duplicates retained. This is the only result reported as the protocol outcome.
- **Secondary sensitivity view (not a replacement):** within each split, rows are grouped by `audio_sha256`, and only the row with the lexicographically smallest `relative_path` is kept. That leaves 3,692 validation rows and 3,953 test rows. The view is computed from the same prediction files with the same `T`, with no new inference and no refit, and is reported separately under `duplicate_sensitivity`.
- **Limitation:** duplicate copies weight some utterances more than once in the official numbers. Near-duplicates (re-encoded or trimmed copies) are not detected.

## 8. Outputs

The evaluation run directory, `runs/<UTC time>_<name>_<suffix>/`, is git-ignored and contains:
- `config.yaml` and `metadata.json` from `start_run`;
- `predictions/validation.csv` and `predictions/test.csv`: one row per manifest row, in manifest order. Columns: `relative_path`, `audio_sha256`, `speaker_id`, `label`, `label_index`, `pred_index`, `pred_label`, `correct`, 10 `logit_<label>` columns (exact float32), and 10 `prob_<label>` columns (raw `T = 1`, float64). Reading a file back re-derives the argmax, correctness, and raw probabilities from the logits and rejects any mismatch.
- `predictions/validation_calibrated.csv` and `predictions/test_calibrated.csv`: `relative_path`, `pred_index`, and 10 `cal_prob_<label>` columns (`softmax(logits / T)`). They are written only after `calibration.json`, because the validation predictions have to exist before `T` can be fitted;
- `calibration.json`;
- `report.json` (every metric, per-class values, confusion matrices, ECE bins, duplicate sensitivity, checkpoint and manifest identity, and the SHA-256 of every artifact) and `report.md`;
- `summary.json`, with `status` completed or failed.

Every file is written atomically, via a temporary file and a rename.
