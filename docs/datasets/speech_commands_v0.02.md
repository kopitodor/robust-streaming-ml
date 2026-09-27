# Verification record: Speech Commands v0.02

Fills in [data contract](../data-contract.md) section 1 from the actual download. Contract: [ADR 0001](../decisions/0001-dataset-speech-commands-v2.md). Every value here was observed or computed on 2026-09-27; none are copied from memory.

The archive lives in the git-ignored folder `data/raw/speech_commands_v0.02.tar.gz`. The top-level files inspected below were extracted to `data/raw/inspect_v0.02/`, also git-ignored. No audio has been decoded yet.

## Section 1 fields

| Field | Value |
| --- | --- |
| `dataset_name` | `speech_commands` |
| `dataset_version` | `v0.02` (the archive's README title: "Speech Commands Data Set v0.02") |
| `source_url` | `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` (the "original location" named in the archive README) |
| `retrieval_url` | `https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz` (see "Retrieval" below) |
| `archive_filename` | `speech_commands_v0.02.tar.gz` |
| `archive_sha256` | `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58` |
| `archive_size_bytes` | `2428923189` |
| `retrieved_at_utc` | `2026-09-27T18:45:33Z` (download completed) |
| `license_name` | Creative Commons Attribution 4.0 International (CC BY 4.0). The `LICENSE` file's first line is "Attribution 4.0 International", and it contains the full "Creative Commons Attribution 4.0 International Public License" text. |
| `license_url` | `https://creativecommons.org/licenses/by/4.0/` (as linked in the archive README) |
| `license_file_path` | `LICENSE` (top level of the archive) |
| `license_file_sha256` | `fd3a263fe19ed8faa9068b43abaebafc02c77897b0c6fc09abc04bb592e5f16e` (18651 bytes) |
| `attribution_text` | See "Attribution" below. |
| `redistribution_notes` | CC BY 4.0 allows redistribution with attribution and an indication of changes. This project redistributes **no audio**. The committed manifest (paths, hashes, audio metadata) will carry the attribution below. |
| `license_verified_by` / `license_verified_on` | **Project owner, 2026-09-27.** The owner approved using Speech Commands v0.02 under CC BY 4.0 for this portfolio project and accepted the attribution requirements recorded below. Claude (AI assistant) read `LICENSE` and `README.md` on the same date and prepared this record. |

Other top-level files, with SHA-256 values computed locally:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `README.md` | 8093 | `b415da1f2d880bcce33b2958fcba9bba46013c1bc5592fa4aaa85759d9308b6f` |
| `validation_list.txt` | 268262 | `5747407275538b4056e823982f0db1fc993776ab532048196a19be701bdc87d2` |
| `testing_list.txt` | 295833 | `2d17c6b3faf63be43eda93cfeb0c747cfd79b7b236282039dbac65a2cb5f1df5` |
| `.DS_Store` | 6148 | not hashed; macOS metadata file, ignored |

## Retrieval

- **HTTP:** `http://download.tensorflow.org/...` returned HTTP 200 with the same object metadata as below. The file wasn't downloaded over plain HTTP.
- **HTTPS on the canonical host:** `https://download.tensorflow.org/...` failed certificate validation: curl `SEC_E_WRONG_PRINCIPAL`, "The target principal name is incorrect".
- **Actual download path:** the same Google Cloud Storage object over HTTPS at `storage.googleapis.com`. Its response headers match the canonical URL exactly: `ETag "6b74f3901214cb2c2934e98196829835"`, `x-goog-generation 1523475715344020`, `Content-Length 2428923189`, `Last-Modified Wed, 11 Apr 2018 19:41:55 GMT`, `x-goog-hash md5=a3TzkBIUyywpNOmBloKYNQ==`.
- **Integrity:** the downloaded file's MD5 (`6b74f3901214cb2c2934e98196829835`, base64 `a3TzkBIUyywpNOmBloKYNQ==`) matches the server's MD5 and ETag, and its size matches `Content-Length`. The SHA-256 above is now the pinned value; later acquisitions must match it.

## Attribution

Required by CC BY 4.0 and requested by the dataset README. It must appear with any published results or redistributed metadata.

> Speech Commands Data Set v0.02, by Pete Warden (Google), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz. Changes: only the 10-keyword subset is used, and it is described by a derived manifest; no audio is redistributed.

The citation requested in the README:

```bibtex
@article{speechcommandsv2,
   author = {{Warden}, P.},
    title = "{Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1804.03209},
 primaryClass = "cs.CL",
     year = 2018,
    month = apr,
    url = {https://arxiv.org/abs/1804.03209},
}
```

(The README's `keywords` line is omitted.)

## Structural cross-checks against the contract

These come from the archive's file listing and the two list files only; no audio was decoded. They don't replace the validation step still to be implemented (contract sections 2–6).

- Archive members: 105,841 regular files and 37 directories.
- Top level: 36 folders (35 words plus `_background_noise_`) and the files listed above.
- `.wav` files in the 35 word folders: **105,829**, matching the README's stated total for v0.02. `_background_noise_/` holds 6 `.wav` files and a `README.md`.
- `validation_list.txt`: 9,981 lines. `testing_list.txt`: 11,005 lines. For both: LF line endings, no empty lines, no backslashes, no duplicates, all 35 words present, and every listed path exists in the archive. No path is in both lists.
- Every word `.wav` filename matches `<hex speaker id>_nohash_<int>.wav`.
- 10-keyword subset, split by the contract's rules:

| Label | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| yes | 3228 | 397 | 419 |
| no | 3130 | 406 | 405 |
| up | 2948 | 350 | 425 |
| down | 3134 | 377 | 406 |
| left | 3037 | 352 | 412 |
| right | 3019 | 363 | 396 |
| on | 3086 | 363 | 396 |
| off | 2970 | 373 | 402 |
| stop | 3111 | 350 | 411 |
| go | 3106 | 372 | 402 |
| **Total** | **30769** | **3703** | **4074** |

  That's 38,546 files from 2,492 distinct speakers, and **no speaker appears in more than one split**.
- The README describes the split rule: a SHA-1 of the file's basename with everything from `_nohash_` onward removed. Because this depends on the speaker ID alone, all of a speaker's clips land in the same split. This matches the contract's premise, and the speaker check above is consistent with it.

**Not yet verified:** sample rate, channel count, sample format, clip durations, all-zero audio, and content-level duplicates (SHA-256 of each file). The README says clips are 16-bit little-endian PCM, 16 kHz, trimmed to one second, but the contract requires these to be checked by decoding every file in the validation step.
