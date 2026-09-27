# Data contract: Speech Commands v2, 10-keyword subset

Status: **planning.** Nothing has been downloaded. Statements marked *expected* describe the dataset as documented upstream and must be confirmed against the downloaded archive; if they are wrong, the acquisition code must fail rather than adapt silently. Decision: [ADR 0001](decisions/0001-dataset-speech-commands-v2.md).

Paths are POSIX-style and relative to the extracted archive root. "Error" means validation fails and no manifest is written; "warning" means the manifest is written and the issue appears in the validation report.

## 1. Source, version, and license

Canonical source (expected):

| Field | Value |
| --- | --- |
| `dataset_name` | `speech_commands` |
| `dataset_version` | `v0.02` |
| `source_url` | `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |
| `archive_filename` | `speech_commands_v0.02.tar.gz` |
| `citation` | Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209 |

Only this archive is used. The separate `speech_commands_test_set_v0.02.tar.gz` is out of scope (ADR 0001).

**Verify before use.** Acquisition records all of these; it must not continue while any required field is empty.

| Field | Required | How it is filled |
| --- | --- | --- |
| `archive_sha256` | yes | SHA-256 of the downloaded archive, computed locally. On re-acquisition, a mismatch is an error. |
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

`LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]`; `label_index` is the position in this list. The label of a file is the name of its top-level directory.

- Files in the ten label directories are candidates. Every other top-level directory, including the other 25 words and `_background_noise_/`, is ignored and is not an error. The report counts the ignored files per directory.
- Error: any of the ten directories is missing or has no `.wav` files.
- Error: a manifest row whose `label` is not in `LABELS`, or whose `label_index` does not match its label.
- Error: any of the ten labels has zero examples in any split.

## 3. Splits

The archive contains `validation_list.txt` and `testing_list.txt` (expected). Each line is one relative path, for example `yes/<speaker>_nohash_<n>.wav`. Both lists cover all 35 words.

1. Record the SHA-256 of both list files in the metadata sidecar.
2. Parse lines: strip whitespace, skip empty lines, and normalise to `/`. Errors: absolute paths, `..`, and duplicate lines within a list.
3. Keep only entries whose label directory is in `LABELS`.
4. Error: a kept entry whose file does not exist.
5. Error: a path that appears in both lists.
6. `split = "validation"` or `"test"` from the list; **`split = "train"` for every other `.wav` file in the ten label directories.** Train is never read from a list and never re-split.
7. No re-sampling, re-balancing, or re-stratification. Per-split, per-label counts are reported as observed.

The upstream lists assign files to splits by a hash of the speaker ID, so speakers should not cross splits. This is verified, not assumed (section 6).

## 4. Audio identity

| Field | Type | Definition |
| --- | --- | --- |
| `relative_path` | str | Canonical ID: POSIX path relative to the archive root. Unique. |
| `speaker_id` | str | Filename prefix before `_nohash_`. Error if the filename does not match `<speaker_id>_nohash_<int>.wav`. |
| `utterance_index` | int | Integer after `_nohash_`. |
| `sha256` | str | SHA-256 of the file's bytes (64 lowercase hex characters). |
| `size_bytes` | int | File size. |

## 5. Audio format and duration

Values are read from the decoded WAV header and data, not from the filename or file size.

| Field | Type | Rule |
| --- | --- | --- |
| `sample_rate_hz` | int | Must be 16000 (expected); otherwise error. No resampling at acquisition. |
| `num_channels` | int | Must be 1; otherwise error. |
| `sample_format` | str | Must be `PCM_16` (expected); otherwise error. |
| `num_frames` | int | Must be at least 1; zero is an error. More than 16000 frames (over 1.000 s at 16 kHz) is an error. |
| `duration_s` | float | `num_frames / sample_rate_hz`, rounded to 6 decimals. |
| `is_all_zero` | bool | True if every sample is 0. Such files are a warning (kept, counted). |

Unreadable or corrupt files are errors. Clips shorter than 1 s are expected and valid. Padding and cropping belong to the Milestone 3 feature policy, not to acquisition. The report shows the duration distribution per split and label: minimum, 5th percentile, median, and the count under 1 s.

## 6. Duplicate and leakage checks

| Check | Result |
| --- | --- |
| Same `relative_path` twice | error |
| Same `sha256` in different splits | error (leakage) |
| Same `sha256` with different labels | error |
| Same `sha256` more than once within one split and label | warning. All rows kept; the report lists each group. |
| `speaker_id` present in more than one split | error (speaker leakage) |

Near-duplicates (the same audio re-encoded, trimmed, or with a gain change) are **not** detected. This limitation must be documented alongside results.

## 7. Manifest schema

Two files, both written together:

- `manifests/speech_commands_v0.02_kw10.csv`: UTF-8, comma-separated, header row, `\n` line endings, one row per file in the ten labels, sorted by `relative_path`. Columns in this order:

  `relative_path, split, label, label_index, speaker_id, utterance_index, sha256, size_bytes, sample_rate_hz, num_channels, sample_format, num_frames, duration_s, is_all_zero`

  `split` ∈ {`train`, `validation`, `test`}; `is_all_zero` ∈ {`true`, `false`}.

- `manifests/speech_commands_v0.02_kw10.json`: metadata sidecar with
  - `manifest_schema_version` (starting at 1), `manifest_sha256` (of the CSV bytes), and `row_count`;
  - every field in section 1 and `LABELS`;
  - `validation_list_sha256` and `testing_list_sha256`;
  - per-split, per-label counts; the warning and ignored-file summaries;
  - `created_at_utc` and the code provenance (`source.commit` and `source.dirty`) recorded the same way as `start_run` records it.

The manifest is deterministic. Running it twice on the same archive must produce byte-identical CSV files (only `created_at_utc` and provenance in the sidecar may differ). It is committed to the repository and it contains no audio. Training and evaluation read splits only from the manifest. Audio under `data/` stays git-ignored.
