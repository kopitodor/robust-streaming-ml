# 0001 — Dataset: Google Speech Commands v2, 10-keyword subset

- **Status:** Accepted. The source, checksum, license, and attribution were verified against the downloaded archive on 2026-09-27 ([verification record](../datasets/speech_commands_v0.02.md)). Human sign-off on the license is still pending.
- **Date:** 2026-09-27
- **Milestone:** 2 (dataset acquisition and validation), planning only

## Context

DESIGN.md requires one public speech or audio dataset with a license that allows local portfolio use and recorded provenance. It also requires deterministic, leakage-checked splits. The baseline is a small log-mel CNN evaluated on short streaming windows with CPU inference.

## Decision

Use **Google Speech Commands, version 2 (archive version `v0.02`)**, restricted to exactly these ten labels, in this order:

| `label_index` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `label` | `yes` | `no` | `up` | `down` | `left` | `right` | `on` | `off` | `stop` | `go` |

Only these ten labels are classified. The dataset's validation and test lists define those two splits. The training split is everything else among the ten labels (see the [data contract](../data-contract.md)).

### Out of scope for the first baseline

- **No `_silence_` class.** No silence examples are generated, including from `_background_noise_/`.
- **No `_unknown_` class.** The other 25 words in v2 are excluded entirely. They are not grouped into an "unknown" label.
- **No separately distributed v2 test-set archive.** That archive contains `_silence_` and `_unknown_` examples. The first baseline uses the test list inside the main archive only.
- **`_background_noise_/` is not used.** It is kept out of every split. Any later use, for example as a noise source in Milestone 5, needs its own decision that covers leakage and licensing.

Consequence: the baseline is a **closed-set** classifier. Its results do not measure rejection of out-of-vocabulary speech or silence, and no reported metric may suggest they do. This limitation must be restated wherever results appear.

## Rationale

- Public, widely used, and documented in a citable technical report (Warden, 2018, arXiv:1804.03209). It is expected to be under a permissive attribution license; this must be verified.
- Short (about 1 s), single-channel, 16 kHz clips fit a small CNN, fixed streaming windows, and CPU latency measurement.
- The official split lists are assigned per speaker. Using them keeps results comparable with other work and gives speaker-disjoint splits that can be checked.
- The ten keywords are the standard subset. They give a balanced, understandable baseline without the extra choices that `_unknown_` and `_silence_` would require.

## Alternatives considered

- **All 35 words:** more classes and more imbalance with little extra portfolio value for a first baseline. Deferred.
- **The 12-class task (10 words + `_silence_` + `_unknown_`):** needs extra policies for building those two classes and weighting them. Deferred until the closed-set baseline is reproducible.
- **Speech Commands v1 (`v0.01`):** older and smaller. v2 is the current release.
- **Custom re-split:** would lose comparability and risk speaker leakage. Rejected in favour of the official lists, verified by our own checks.

## Consequences

- The data contract fixes the manifest schema and the validation rules before any code is written.
- Before acquisition code runs, the archive checksum, the license text, and the attribution must be recorded, as values actually observed.
- DESIGN.md's generic requirement for class-stratified splits is resolved by using the official speaker-based lists (see DESIGN.md, ADR-001).
