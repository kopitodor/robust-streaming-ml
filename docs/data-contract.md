# Data contract: Speech Commands v2, 10-keyword subset

Status: **implemented and passed on 2026-09-27.** Section 1 was verified against the downloaded archive. Sections 2–7 are implemented in `robust_streaming_ml.data.validation` and `robust_streaming_ml.data.manifest`, run by `scripts/build_speech_commands_manifest.py`. Results are in the [verification record](datasets/speech_commands_v0.02.md). Changes from the planning version are listed under [Amendments](#amendments). Statements marked *expected* describe the dataset as documented upstream and must be confirmed against the downloaded archive; if they are wrong, the acquisition code must fail rather than adapt silently. Decision: [ADR 0001](decisions/0001-dataset-speech-commands-v2.md).

Paths are POSIX-style and relative to the extracted archive root. "Error" means validation fails and no manifest is written; "warning" means the manifest is written and the issue appears in the validation report.

## 1. Source, version, and license

Canonical source (expected):

| Field | Value |
| --- | --- |
| `dataset_name` | `speech_commands` |
| `dataset_version` | `v0.02` |
| `source_url` | `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |
| `archive_filename` | `speech_commands_v0.02.tar.gz` |
| `retrieval_url` | `https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz`: the same object as `source_url`, over HTTPS (HTTPS on `download.tensorflow.org` fails certificate validation). |
| `archive_sha256` (pinned) | `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58` |
| `archive_size_bytes` (pinned) | `2428923189` |
| `citation` | Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209 |

Only this archive is used. The separate `speech_commands_test_set_v0.02.tar.gz` is out of scope (ADR 0001).

**Verify before use.** Acquisition records all of these; it must not continue while any required field is empty.

| Field | Required | How it is filled |
| --- | --- | --- |
| `archive_sha256` | yes | SHA-256 of the downloaded archive, computed locally. It must equal the pinned value above; a mismatch is an error. |
| `archive_size_bytes` | yes | Measured locally. |
| `retrieved_at_utc` | yes | Time of download. |
| `license_name` | yes | From the license file shipped in the archive (expected: Creative Commons Attribution 4.0, CC BY 4.0). |
| `license_url` | yes | Canonical URL of that license. |
| `license_file_path` | yes | Path of the license file inside the archive (expected `LICENSE`); its SHA-256 is recorded too. |
| `license_file_sha256` | yes | Computed locally. |
| `attribution_text` | yes | Attribution/citation required by the license and dataset README, copied verbatim. |
| `redistribution_notes` | yes | What may be redistributed. Audio is never committed to this repository. The manifest (paths, hashes, metadata) is committed. |
| `license_verified_by` / `license_verified_on` | yes | The person who read the license text, and the date. This is a human check, not automated. |

## 2. Label set

`LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]`; `label_index` is the position in this list. The label of a file is the name of its top-level directory (exact case).

- Files in the ten label directories are candidates. Every other top-level directory, including the other 25 words and `_background_noise_/`, is ignored and is not an error. The report counts the ignored files per directory. Top-level files other than `LICENSE`, `README.md`, the two split lists, and the acquisition record `.acquisition.json` (for example the archive's `.DS_Store`) are ignored and listed in the report. A top-level symbolic link is an error.
- Error: any of the ten directories is missing, is not a regular directory, or has no valid `.wav` files.
- Error: anything inside a label directory that is not a regular file (a subdirectory or symbolic link), or a file whose name does not match section 4's filename rule. Non-`.wav` files count as bad filenames.
- Error: a manifest row whose `label` is not in `LABELS`, or whose `label_index` does not match its label.
- Error: any of the ten labels has zero examples in any split.

## 3. Splits

The archive contains `validation_list.txt` and `testing_list.txt` (confirmed). Each line is one relative path, for example `yes/<speaker>_nohash_<n>.wav`. Both lists cover all 35 words.

1. The SHA-256 of both list files must equal the pinned values (`ArchiveSpec.pinned_members`), as must `LICENSE` and `README.md`. A missing or changed pinned file is an error. The list hashes are recorded in the sidecar.
2. Parse each list as UTF-8 (anything else is an error). For each line: strip whitespace, skip empty lines, and replace backslashes with `/`. Errors: absolute paths (leading `/` or a drive letter); non-canonical paths (any `.`, `..`, or empty segment); and duplicate lines within a list.
3. Keep only entries whose first path segment is in `LABELS`, and count the others. A kept entry must be exactly `<label>/<filename>`, with the filename matching section 4; otherwise it is an error.
4. Error: a kept entry whose file does not exist among the candidates.
5. Error: a path that appears in both lists.
6. `split = "validation"` or `"test"` from the list; **`split = "train"` for every other candidate in the ten label directories.** Train is never read from a list and never re-split.
7. No re-sampling, re-balancing, or re-stratification. Per-split, per-label counts are reported as observed.

The upstream lists assign files to splits by a hash of the speaker ID, so speakers should not cross splits. This is verified, not assumed (section 6).

## 4. Audio identity

Filename rule: `<speaker_id>_nohash_<utterance_index>.wav`, where `speaker_id` is exactly 8 lowercase hexadecimal characters, `utterance_index` is a canonical non-negative integer (no leading zeros), and the extension is a lowercase `.wav`. Any other name is an error.

| Field | Type | Definition |
| --- | --- | --- |
| `relative_path` | str | Canonical ID: POSIX path `<label>/<filename>` relative to the archive root. Unique, including case-insensitively. |
| `speaker_id` | str | Filename prefix before `_nohash_`. |
| `utterance_index` | int | Integer after `_nohash_`. |
| `sha256` | str | SHA-256 of the file's bytes (64 lowercase hex characters). |
| `audio_sha256` | str | SHA-256 of the decoded audio content (section 6 protocol). |
| `size_bytes` | int | File size. |

## 5. Audio format and duration

Values are read by a strict RIFF/WAVE parser (`robust_streaming_ml.data.wav`), not from the filename or file size.

Structural errors (`wav_malformed`):
- no `RIFF`/`WAVE` signature, or a RIFF size different from file size − 8;
- a truncated chunk, a chunk running past the end of the file, or trailing bytes;
- a missing, duplicated, or misordered `fmt ` or `data` chunk (`fmt ` must come first);
- a `fmt ` chunk shorter than 16 bytes, or `block_align` and `byte_rate` inconsistent with the channels, bit depth, and sample rate;
- a data size that is not a whole number of frames.

Other chunks are allowed and are counted in the report. Unreadable files are errors.

| Field | Type | Rule |
| --- | --- | --- |
| `sample_rate_hz` | int | Must be 16000; otherwise error. No resampling. |
| `num_channels` | int | Must be 1; otherwise error. |
| `sample_format` | str | Must be `PCM_16` (format tag 1, 16 bits); otherwise error. This includes WAVE_FORMAT_EXTENSIBLE and float formats. |
| `num_frames` | int | Must be at least 1; zero is an error. More than 16000 frames (over 1.000 s at 16 kHz) is an error. |
| `duration_s` | str | `num_frames / sample_rate_hz` formatted with 6 decimals (`f"{x:.6f}"`). |
| `peak_abs` | int | Largest absolute sample value, 0–32768. |
| `is_all_zero` | bool | `peak_abs == 0`. Warning `all_zero_audio` (kept, counted). |

A clip with `0 < peak_abs <= 32` (at or below about −60 dBFS) is a warning, `near_silent_audio` (kept, counted). Clips shorter than 1 s are expected and valid. Padding and cropping belong to the Milestone 3 feature policy. The report shows the duration distribution per split and label: count, minimum, 5th percentile, median, and maximum (nearest-rank), and the count under 16,000 frames.

## 6. Duplicate, leakage, and integrity checks

**Content-hash protocol.** `sha256` hashes the whole file. `audio_sha256` hashes only the bytes of the `data` chunk: little-endian signed 16-bit mono samples, excluding the RIFF and `fmt ` headers, any other chunks, and the pad byte. Two files with identical samples but different headers therefore share an `audio_sha256`. The duplicate rules use `audio_sha256`, so they cover byte-identical files as well.

| Check | Result |
| --- | --- |
| Same `relative_path` twice, identical rows, or paths differing only by case | error |
| Same `audio_sha256` in different splits | error (leakage) |
| Same `audio_sha256` with different labels | error |
| Same `audio_sha256` more than once within one split and label | warning. All rows kept; the report lists each group with its speakers and counts groups per split and groups spanning more than one speaker. |
| `speaker_id` present in more than one split | error (speaker leakage) |
| A row's path inconsistent with its label, speaker, or index, or out-of-range fields (bad hash format, format fields, duration, `peak_abs`) | error (independent row checks in `check_rows`) |

Near-duplicates (the same audio re-encoded, trimmed, or with a gain change) are **not** detected. This limitation must be documented alongside results.

**Archive cross-check.** The pinned archive is streamed during validation, and its SHA-256 and size must match again. Every regular file in the ten label directories must exist in both the archive and the extraction with identical bytes. Errors: `archive_checksum_mismatch`, `archive_unreadable`, `extraction_missing_file`, `extraction_extra_file`, `extraction_modified_file`. The extraction must also carry an `.acquisition.json` record for the pinned archive; otherwise validation does not start.

## 7. Manifest and reports

Outputs of `scripts/build_speech_commands_manifest.py`, all under the git-ignored `data/` folder:

- `data/manifests/speech_commands_v0.02_kw10.csv`: UTF-8, comma-separated, header row, LF line endings, one row per file in the ten labels, sorted by `relative_path` (Python string order). Booleans are written `true`/`false` and integers in canonical decimal. Columns in this order:

  `relative_path, split, label, label_index, speaker_id, utterance_index, sha256, audio_sha256, size_bytes, sample_rate_hz, num_channels, sample_format, num_frames, duration_s, peak_abs, is_all_zero`

  `split` ∈ {`train`, `validation`, `test`}. `read_manifest` decodes the file strictly (re-encoding must reproduce the bytes) and can check it against an expected SHA-256.

- `data/manifests/speech_commands_v0.02_kw10.json`: metadata sidecar with
  - `manifest_schema_version` (1), `manifest_file`, `manifest_sha256` (of the CSV bytes), `row_count`, and `columns`;
  - every field in section 1 (archive, license, attribution, citation, and verification sign-off) and `LABELS`;
  - `validation_list_sha256` and `testing_list_sha256`;
  - per-split and per-label counts, speaker counts, and the warning and ignored-file summaries;
  - `content_hash_protocol`, and the validation report's file name and SHA-256;
  - `created_at_utc` and the code provenance (`source.commit` and `source.dirty`), resolved as `start_run` does. If the revision cannot be resolved, nothing is written.

- `data/reports/speech_commands_v0.02_kw10_validation.json` (machine-readable) and `.md` (human-readable): status, contract parameters, counts, speakers, split-list statistics, duration statistics, duplicate groups, archive cross-check, ignored items, and every error and warning with a stable `code`. They contain no timestamps or absolute paths.

**Determinism.** The CSV and both reports are byte-identical across runs on the same data and code; only `created_at_utc` and the provenance in the sidecar may differ. The CSV's SHA-256 is **pinned** in `SPEECH_COMMANDS_KW10.expected_manifest_sha256`. A generated manifest with any other hash is an error (`manifest_checksum_mismatch`), so any change in code or data that alters the manifest must be deliberate and re-pinned. Training and evaluation read splits only from a manifest whose SHA-256 matches the pin.

**Fail closed.** On any error, no manifest or sidecar is written, and existing ones at those paths are deleted. The reports are still written with `status: failed` and every error listed, and the command exits non-zero. Nothing is repaired, relabelled, omitted, or resampled.

## Amendments

Changes made on 2026-09-27 when implementing sections 2–7, relative to the planning version:

1. **Manifest location:** the manifest was planned as a committed file under `manifests/`. It is now generated under the git-ignored `data/manifests/`, and only its SHA-256 is committed (pinned in code and in the verification record), per the project owner's instruction to commit only code, tests, and docs. The manifest is deterministic from the pinned archive and the pinned code, so its traceability is unchanged.
2. **New columns:** `audio_sha256`, for header-independent duplicate detection, and `peak_abs`, for silence reporting. The duplicate rules in section 6 now apply to `audio_sha256` instead of the file `sha256`.
3. **Stricter rules, matching the observed data:** 8-hex-character speaker IDs; canonical utterance indices; strict WAV structure (section 5); no non-file entries in label directories; canonical split-list paths.
4. **Near-silent warning** (`0 < peak_abs <= 32`), in addition to the all-zero warning.
5. **Archive cross-check** of every label-directory file, and a **pinned manifest SHA-256**.
