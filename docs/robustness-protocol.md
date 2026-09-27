# Robustness protocol `kw10-robustness-v1`

Milestone 5. This protocol and its implementation (`robust_streaming_ml.robustness`) were fixed and committed **before** any robustness evaluation of the real test split was run. It is run by `scripts/evaluate_robustness.py` with [`configs/kw10_cnn_baseline_robustness.yaml`](../configs/kw10_cnn_baseline_robustness.yaml). Results go in [RESULTS.md](../RESULTS.md).

## 1. Fixed inputs

| Item | Value |
| --- | --- |
| Checkpoint | `runs/20260927T210911Z_kw10-cnn-baseline_6c9d6cdf/checkpoints/best.pt`, SHA-256 `d21f89283a128820e93b00d6e104805b3423fd47f2ed8791ba7775d5bf9d267e` (Milestone 3, epoch 18). This is the checkpoint evaluated in Milestone 4. It is **not** retrained, tuned, or recalibrated. |
| Manifest | `speech_commands_v0.02_kw10.csv`, SHA-256 `f811a20d…ab6131c` |
| Labels | `yes, no, up, down, left, right, on, off, stop, go`, in this order |
| Preprocessing | the checkpoint's window (version 1, `start_zero_pad`, 16,000 samples) and log-mel features (version 1, fingerprint `8368ac49…00ba1`) |
| Split | the official **test** split only: all 4,074 rows, duplicates retained |
| Probabilities | **raw** softmax (`T = 1`). The Milestone 4 temperature is not applied and nothing is refitted. |
| Clean baseline | the Milestone 4 official raw test metrics from run `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a`, whose `report.json` is pinned by SHA-256 `2f9a0f5bdbbdb89cf1053c604e821ccd139950a73d6fe07ab4e059872d56b237`. Accuracy 0.9639175, macro F1 0.9638840, NLL 0.1010780, ECE 0.0082416. |
| Inference | eval mode, CPU, float32, batch size 256 and 8 torch threads, both equal to the baseline's and checked |

## 2. Gates, checked before anything is written

1. The checkpoint gate from Milestone 4: the pinned SHA-256 and sidecar; compatibility and the feature fingerprint; a clean training tree whose commit is an ancestor of HEAD; agreement with the training-run metadata.
2. The validated-manifest gate.
3. **The baseline:** the pinned `report.json` SHA-256. Its protocol must be `kw10-clean-eval-v1`, and its checkpoint, manifest, labels, batch size, and thread count must equal this run's. Its `summary.json` status must be `completed`, and `predictions/test.csv` must match the hash in the report.
4. **The codec backend is exactly the pinned one:** `soundfile` 0.14.0 and libsndfile 1.2.2, with Ogg Opus support. Anything else stops the run with a diagnostic. A missing or different backend is never replaced, skipped, or weakened.
5. A clean working tree (`--allow-dirty-source` is for development only).

**Clean-path reproduction.** Inside the run, before any corruption, the robustness inference path runs on the unmodified test clips. It must reproduce the Milestone 4 raw test logits **bit-for-bit**, otherwise the run fails. So every corrupted metric comes from exactly the pipeline that produced the clean baseline.

## 3. Corruptions (all version 1)

Each corruption maps one clean clip to a clip of exactly the same length. The input is int16 PCM with its own `n` frames, 16 kHz, mono; the output is float32 at 16 kHz. The fixed window (zeros appended to 16,000 samples) is applied afterwards, as on the clean path. Arithmetic is float64 until a final cast to float32, with `x = int16 / 32768`. Outputs are **not** clipped or rescaled to full scale: the model receives the float values as they are, and samples above 1.0 in magnitude are counted.

| Corruption | Transform | Severities (mild → severe) |
| --- | --- | --- |
| `noise` | Add white Gaussian noise at an **exact** SNR relative to the clip's mean power: `w` is drawn standard normal, rescaled to exactly unit mean power, and `y = x + sqrt(P_x / 10^(SNR/10)) · w`. | `snr_20db`, `snr_10db`, `snr_0db` (SNR 20, 10, 0 dB) |
| `reverb` | Convolve with a synthetic room impulse response (RIR). Its lag 0 is 1 (the direct path). Lags `1 … round(RT60·16000) − 1` are a Gaussian tail whose amplitude decays by 60 dB over RT60 (`·10^(−3k/(RT60·fs))`), scaled so that tail energy is `10^(−DRR/10)`. The convolution uses a zero-padded FFT, the output is truncated to `n` frames, and it is rescaled to the input RMS. | `rt60_0.3s_drr_+6db`, `rt60_0.6s_drr_0db`, `rt60_1.0s_drr_-6db` |
| `resample` | Resample 16 kHz → intermediate rate → 16 kHz with a lowpass FIR: `32·M + 1` taps, `h = sinc(m/M)/M · kaiser(β = 8)`, normalised to unit DC gain, where `M = 16000 / rate`. Downsampling filters (centred, zero edges) and keeps every M-th sample; upsampling inserts zeros, multiplies by M, filters, and truncates to `n`. | `via_8000hz`, `via_4000hz`, `via_2000hz` (M = 2, 4, 8) |
| `clipping` | Hard saturation at `c = f · peak(x)`, then divide by `f`. This is equivalent to gain `1/f` followed by saturation at the clip's own peak, so the peak level is preserved and only the waveform is distorted. | `at_0.5_peak`, `at_0.25_peak`, `at_0.1_peak` (f = 0.5, 0.25, 0.1) |
| `codec` | Ogg Opus via `soundfile` 0.14.0 / libsndfile 1.2.2. The int16 clip is encoded to an in-memory Ogg Opus stream (16 kHz, mono) with `compression_level` L, then decoded with `soundfile.read(dtype="float32")`. The decoded rate must be 16,000 Hz, the channel count 1, and the frame count exactly `n` (libsndfile handles Opus pre-skip; on synthetic probes the lag was 0), or it is an error. The encoded size is recorded per clip as kbit per second of audio, including container overhead. | `opus_level_0.9`, `opus_level_0.95`, `opus_level_1.0` (L = 0.9, 0.95, 1.0; on a synthetic 1 s probe during implementation this was about 38, 25, and 13 kbit/s) |

**Inputs are procedural only.** No external recordings are used. The dataset's `_background_noise_/` files are deliberately not used (ADR 0001).

**Codec licensing** (backend only; nothing is redistributed): `soundfile` is BSD-3-Clause. It bundles libsndfile (LGPL-2.1+), which uses libopus and libogg (BSD-3-Clause). Encoded Ogg bytes are not bit-reproducible, because libsndfile writes a random Ogg stream serial number (bytes 14–17 and the page CRC at 22–25). The **decoded samples** are deterministic, and those are what the protocol uses and hashes.

## 4. Seeds and determinism

- `seed = derive_seed(1729, "kw10-robustness-v1", corruption, version, severity, relative_path)`, i.e. the first 8 bytes of `SHA-256("1729|kw10-robustness-v1|<corruption>|1|<severity>|<relative_path>")`, modulo 2^32. The random generator is `numpy.random.default_rng(seed)` (PCG64). `noise` and `reverb` use it; `resample`, `clipping`, and `codec` are deterministic and record the seed anyway.
- Each clip's corruption depends only on its identity and the protocol, not on processing order, batching, or the number of `corruption_workers` (tested).
- `corruption_manifest.csv` records, per clip: `relative_path`, source `audio_sha256`, corruption, version, severity, seed, `output_sha256` (the SHA-256 of the little-endian float32 output), and the measurements.

## 5. Output validation

Every corrupted clip must be:
- float32, exactly `n` frames (so it fits the window);
- entirely finite;
- **different from the clean input**: a transform that changed nothing is an error, not a silent weakening. An all-zero clip is an error for noise, reverb, and clipping (undefined SNR or peak) and for resample (no change). Opus decodes digital silence to a floor of at most 1 LSB. The test split has no all-zero clips.

The codec output's rate, channel count, and length are checked; no other resampling or repair happens outside the declared transforms. The source audio is read-only: each transform receives copies. Generated audio is kept only if `retain_audio_examples > 0`, as float32 WAV under the git-ignored run directory, and it is read back and compared.

## 6. Metrics, degradation, and duplicates

- For every corruption × severity (15 conditions): accuracy, macro F1 (a 0/0 class gets 0), NLL, top-label ECE with 15 equal-width bins, the confusion matrix (rows are the truth, in fixed label order), per-class precision, recall, and F1, and the 5 largest confusions. The definitions are identical to [docs/evaluation-protocol.md](evaluation-protocol.md) §5.
- **Degradation = clean metric − corrupted metric**, at the same split and protocol, for accuracy, macro F1, NLL, and ECE. A positive value means worse for accuracy and macro F1; a negative value means worse for NLL and ECE.
- **Primary result:** all 4,074 test rows, duplicates retained.
- **Secondary only:** within each condition, one row per `audio_sha256` (the smallest `relative_path`). That leaves 3,953 rows, and degradation is measured against the Milestone 4 secondary (dedup) raw test metrics. It never replaces the primary result.
- No corruption, severity, threshold, or setting is chosen using corrupted test results. Every condition in this table is reported, including failures.

## 7. Outputs

The run directory `runs/<UTC time>_kw10-cnn-baseline-robustness_<suffix>/` (git-ignored) contains:
- `config.yaml` and `metadata.json`;
- `conditions/<corruption>/<severity>/predictions.csv`, in the Milestone 4 format (logits, raw probabilities, labels, and correctness per test row), and `corruption_manifest.csv`;
- `report.json`: the protocol description, codec backend versions, numpy version, baseline references, the clean-path check, and per condition the metrics, confusion matrix, degradation, top confusions, measurement summaries, and duplicate sensitivity, plus the SHA-256 of every artifact;
- `report.md` and `summary.json`.

Files are written atomically. A failure after the run starts is recorded as `status: failed` in `summary.json`, and no report is written.

## 8. Known limitations

- The corruptions are **synthetic and controlled**: white noise, not real environments; an exponential-decay RIR, not measured rooms; one resampler design; symmetric hard clipping; and one codec family (Opus) at three `compression_level` settings.
- The results describe this model's sensitivity to these specific transforms. They say nothing about real-world robustness in general.
- Three severities per corruption, one model, and one corruption draw per clip (one seed), so there are no confidence intervals.
- Codec bitrate is controlled indirectly through libsndfile's `compression_level`. The effective bitrate varies per clip and is reported, not fixed.
- Bit-exactness is guaranteed only for the same platform and library versions. NumPy's generator stream and the FFT and Opus implementations may differ on other CPUs or versions, which are recorded.
- The duplicate and near-duplicate limitations of Milestone 4 carry over. Results cover clean-trained, closed-set, 1 s clips only: no streaming, no silence or unknown class.
