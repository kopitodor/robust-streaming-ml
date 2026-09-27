# Baseline: log-mel CNN keyword classifier

Milestone 3. The code lives in `robust_streaming_ml.features`, `robust_streaming_ml.model`, `robust_streaming_ml.checkpoint`, and `robust_streaming_ml.training`. The configuration is [`configs/kw10_cnn_baseline.yaml`](../configs/kw10_cnn_baseline.yaml), and the entry point is `scripts/train_baseline.py`. Measured runs are recorded in [RESULTS.md](../RESULTS.md).

The model is a **closed-set** classifier over the 10 keywords (ADR 0001). It has no silence or unknown class.

## Inputs and compatibility gate

Training reads **only** the validated manifest. Before anything is written, `load_validated_manifest` checks that:

1. `data.dataset` names the dataset spec (`speech_commands_v0.02_kw10`), and `data.manifest_sha256` equals the SHA-256 pinned in that spec;
2. `data.labels` equals the spec's label order, since `label_index` is a position in that list;
3. the window sample rate equals the dataset's 16 kHz, and the window is at least the contract's maximum clip length (16,000 samples), so nothing can need cropping;
4. the validation report says `passed` for this manifest and archive, and the manifest sidecar records the report's exact SHA-256;
5. the manifest file hashes to the pinned value, decodes strictly, and passes `check_rows`, with every split non-empty.

Next, the source revision must be clean. A dirty tree is refused unless `--allow-dirty-source` is passed, which is for development runs only and is recorded as `dirty: true`. Only then does `start_run` record provenance, apply seeding, and create the run directory.

Only the **train** and **validation** splits are loaded. Asking for `test` raises an error, and a test checks that no test file is even opened. Every file is re-verified against its manifest row: file SHA-256, strict WAV parse, sample rate, channels, format, and frame count. Any mismatch raises `DataIntegrityError`; nothing is repaired, resampled, or cropped.

## Window and features (versions 1)

| Step | Setting |
| --- | --- |
| Window | `start_zero_pad`: the clip starts at sample 0 and zeros are appended to 16,000 samples (1 s). A longer clip is an error. |
| Scaling | `int16 / 32768` → float32 in [-1, 1) |
| STFT | `n_fft` 512 (32 ms), `win_length` 400 (25 ms, periodic Hann), `hop_length` 160 (10 ms), `center=True` with zero padding → 101 frames |
| Mel | 40 triangular filters on the HTK mel scale from 20 Hz to 8 kHz, peak 1, no area normalisation; computed in float64 and stored as float32 |
| Log | natural `log(mel_power + 1e-6)` |
| Output | `[B, 1, 40, 101]` float32 |

The front end has no trainable parameters. `feature_fingerprint` hashes the settings together with the exact window and filterbank tensors. Checkpoints store the fingerprint, and loading a checkpoint rejects one whose fingerprint differs from what the current code produces.

## Model (`kws_cnn_v1`)

| Layer | Output shape (per example) |
| --- | --- |
| input log-mel | `[1, 40, 101]` |
| `BatchNorm2d(1)` (learned input scaling) | `[1, 40, 101]` |
| block 1: conv3×3 (16) → BN → ReLU → maxpool 2 | `[16, 20, 50]` |
| block 2: conv3×3 (32) → BN → ReLU → maxpool 2 | `[32, 10, 25]` |
| block 3: conv3×3 (64) → BN → ReLU → maxpool 2 | `[64, 5, 12]` |
| block 4: conv3×3 (64) → BN → ReLU → maxpool 2 | `[64, 2, 6]` |
| global average pool | `[64]` |
| dropout 0.1 → linear | `[10]` logits |

**61,052 trainable parameters**:
- input BN: 2;
- convolutions (no bias): 144 + 4,608 + 18,432 + 36,864;
- block BNs: 32 + 64 + 128 + 128;
- linear: 650.

A test pins this count, and another pins the layer shapes.

## Training procedure

- Loss is cross-entropy, using the AdamW optimizer (learning rate 0.001, weight decay 0.0001). The learning rate follows a cosine schedule, stepped per epoch, reaching 0 after `epochs`.
- Batch size is 64, for 30 epochs. The front end runs without gradients inside the training step.
- After each epoch, validation **loss** (mean cross-entropy) and **accuracy** are computed on the validation split in eval mode. The checkpoint with the lowest validation loss is written as `best`; `last` is written every epoch.
- These validation numbers are monitoring values used for checkpoint selection. Formal evaluation (held-out test, macro F1, confusion matrix, calibration) is Milestone 4.
- Augmentation is **off** by default. If enabled explicitly, `time_shift` shifts each training window by a uniform integer number of samples in ±`time_shift_max_ms`, filling with zeros.
- A non-finite loss stops the run. `summary.json` then records `status: failed`.

## Determinism

`start_run` seeds Python, NumPy, and PyTorch, enables `torch.use_deterministic_algorithms`, and records the applied state. Each source of randomness then gets its own derived seed, `derive_seed(seed, purpose, ...)` (the first 8 bytes of a SHA-256 of `"seed|purpose|..."`, modulo 2^32):

| Purpose | Seed |
| --- | --- |
| model initialisation | `derive_seed(seed, "model_init")` |
| dropout (global RNG during training) | `derive_seed(seed, "dropout")` |
| shuffling, epoch `e` | `derive_seed(seed, "shuffle", e)` |
| augmentation, epoch `e`, example `i` | `derive_seed(seed, "augment", e, i)` |
| DataLoader generator (worker base seeds) | `derive_seed(seed, "loader")` |

The sampler yields `(epoch, index)` pairs, so batches, including augmented ones, are identical with 0 or N DataLoader workers (tested). Workers also seed Python and NumPy from their torch seed. Two identical runs give bit-identical weights and logs, apart from wall-clock seconds (tested). The torch thread count is fixed by `training.num_threads` and recorded.

## Checkpoint format (version 1)

`checkpoints/<kind>.pt` is a `torch.save` of `{format, format_version, metadata, model_state}`. It contains only tensors and JSON-compatible values, so it is loaded with `weights_only=True`. `checkpoints/<kind>.json` repeats the metadata and adds the `.pt` SHA-256 and every tensor's name, shape, and dtype.

The metadata records:
- `kind`, `epoch`, `run_id`, `labels`;
- the full `config` and its `config_sha256`;
- `model`: architecture, settings, class count, parameter count, input and layer shapes;
- `preprocessing`: window, features, feature version, fingerprint, input scaling;
- `manifest`: dataset, SHA-256, path, rows per split, splits used;
- `source`: commit and dirty flag;
- `seeding`: the applied state and the derived seeds;
- `validation` loss and accuracy at that epoch;
- `torch_version`.

`load_checkpoint` rejects:
- a wrong format or version, missing metadata, or a sidecar hash mismatch;
- files that are not plain data (arbitrary pickles);
- an unsupported architecture, or a feature version or fingerprint that doesn't match this code;
- a state dict that does not load strictly, or a parameter-count or class-count mismatch;
- a mismatch with any expected manifest SHA-256, labels, window, or features the caller passes.

## Run directory

`runs/<UTC time>_<name>_<suffix>/` (git-ignored) contains:
- `config.yaml` and `metadata.json` (from `start_run`: provenance, applied seeding, environment);
- `train_log.jsonl`: one line per epoch with learning rate, train loss and accuracy, validation loss and accuracy, whether the checkpoint was best, and seconds;
- `checkpoints/{best,last}.{pt,json}`;
- `summary.json`: status, best and last checkpoints with SHA-256, parameter count, layer shapes, rows loaded (test is always 0), manifest SHA-256, fingerprint, derived seeds, threads, and timings.

## Not in this milestone

There is no held-out test evaluation, no calibration or confusion matrix, no robustness corruptions, no streaming inference, no ONNX export, no latency benchmark, and no demo.
