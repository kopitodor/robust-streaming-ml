"""Robustness evaluation entry point (protocol ``kw10-robustness-v1``).

Before anything is written, :func:`evaluate_robustness` checks:

1. the pinned checkpoint and its provenance (the same gate as the Milestone 4 evaluation);
2. the validated manifest;
3. the pinned Milestone 4 baseline: report SHA-256, protocol, checkpoint,
   manifest, labels, batch size, threads, and test predictions;
4. the exact pinned codec backend;
5. a clean source tree.

It then calls ``start_run``. Inside the run it:

1. loads the official test split (every row, duplicates retained), re-verified
   against the manifest;
2. **reproduces the Milestone 4 raw test logits bit-for-bit** through the
   robustness inference path, and fails otherwise, so every degradation is
   measured on the same pipeline;
3. for each of the 5 corruptions x 3 severities: corrupts every clip
   (per-example seeds, any order), validates the outputs, predicts with raw
   probabilities (``T = 1``; no refit), and writes
   ``conditions/<corruption>/<severity>/predictions.csv`` and
   ``corruption_manifest.csv``;
4. writes ``report.json``, ``report.md``, and ``summary.json``.

Clean metrics are the Milestone 4 official raw test metrics.
Degradation = clean metric - corrupted metric, for accuracy, macro F1,
NLL, and ECE: a positive accuracy/F1 degradation and a negative NLL/ECE
degradation both mean worse. The duplicate-aware view is secondary.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from robust_streaming_ml.checkpoint import LoadedCheckpoint
from robust_streaming_ml.data.manifest import ManifestRow
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import DatasetSpec
from robust_streaming_ml.data.wav import encode_wav_float32, parse_wav
from robust_streaming_ml.evaluation.config import ECE_BINS
from robust_streaming_ml.evaluation.metrics import classification_metrics
from robust_streaming_ml.evaluation.predictions import (
    PredictionSet,
    decode_predictions,
    encode_predictions,
    infer_logits,
)
from robust_streaming_ml.evaluation.run import _dedup_indices, verify_checkpoint
from robust_streaming_ml.robustness.config import RobustnessConfig
from robust_streaming_ml.robustness.corruptions import (
    CORRUPTIONS,
    PROTOCOL_ID,
    PROTOCOL_SEED,
    CorruptedClip,
    CorruptionSpec,
    Severity,
    codec_backend_info,
    corrupt,
    output_sha256,
)
from robust_streaming_ml.run_dir import (
    ProvenanceError,
    RunDirectory,
    current_source_revision,
    start_run,
)
from robust_streaming_ml.training.config import WindowSection
from robust_streaming_ml.training.data import (
    ValidatedManifest,
    check_manifest_inputs,
    read_split_audio,
)

__all__ = [
    "BASELINE_PROTOCOL",
    "METRIC_KEYS",
    "RobustnessError",
    "RobustnessResult",
    "degradation",
    "evaluate_robustness",
    "top_confusions",
]

BASELINE_PROTOCOL = "kw10-clean-eval-v1"
METRIC_KEYS = ("accuracy", "macro_f1", "nll", "ece")
REPORT_SCHEMA_VERSION = 1
SPLIT = "test"


class RobustnessError(RuntimeError):
    """A gate failed or the robustness run cannot be completed as specified."""


@dataclass(frozen=True)
class Baseline:
    run_id: str
    report_sha256: str
    clean_raw: dict[str, float]
    clean_dedup_raw: dict[str, float]
    test_predictions: PredictionSet
    test_predictions_sha256: str


@dataclass(frozen=True)
class RobustnessResult:
    run: RunDirectory
    report: dict[str, Any]
    summary: dict[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return _sha256(data)


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def degradation(clean: dict[str, float], corrupted: dict[str, Any]) -> dict[str, float]:
    """``clean - corrupted`` for each metric (protocol definition)."""
    return {key: float(clean[key]) - float(corrupted[key]) for key in METRIC_KEYS}


def top_confusions(metrics: dict[str, Any], count: int = 5) -> list[dict[str, Any]]:
    """Largest off-diagonal confusion counts; ties broken by label order."""
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["rows_true_cols_pred"]
    cells = [
        (-matrix[i][j], i, j)
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and matrix[i][j] > 0
    ]
    return [
        {"true": labels[i], "predicted": labels[j], "count": -neg}
        for neg, i, j in sorted(cells)[:count]
    ]


def _check_baseline(
    config: RobustnessConfig, loaded: LoadedCheckpoint, manifest: ValidatedManifest
) -> Baseline:
    run_dir = config.baseline.run_dir
    try:
        report_bytes = (run_dir / "report.json").read_bytes()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise RobustnessError(f"baseline evaluation run unreadable: {exc}") from exc
    digest = _sha256(report_bytes)
    if digest != config.baseline.report_sha256:
        raise RobustnessError(
            f"baseline report SHA-256 {digest} != pinned {config.baseline.report_sha256}"
        )
    report = json.loads(report_bytes)
    checks = {
        "protocol": (report["protocol"]["id"], BASELINE_PROTOCOL),
        "checkpoint": (report["checkpoint"]["sha256"], loaded.sha256),
        "manifest": (report["manifest"]["sha256"], manifest.sha256),
        "labels": (report["labels"], list(manifest.labels)),
        "status": (summary.get("status"), "completed"),
        "batch_size": (report["batch_size"], config.robustness.batch_size),
    }
    if config.robustness.num_threads is not None:
        checks["torch_num_threads"] = (report["torch_num_threads"], config.robustness.num_threads)
    for name, (found, expected) in checks.items():
        if found != expected:
            raise RobustnessError(f"baseline {name} {found!r} != expected {expected!r}")
    test_bytes = (run_dir / "predictions" / "test.csv").read_bytes()
    test_sha = _sha256(test_bytes)
    if test_sha != report["artifacts_sha256"]["predictions/test.csv"]:
        raise RobustnessError("baseline test predictions do not match the baseline report")
    predictions = decode_predictions(test_bytes, split=SPLIT, labels=manifest.labels)
    raw = report["metrics"][SPLIT]["raw"]
    dedup = report["duplicate_sensitivity"][SPLIT]["raw"]
    return Baseline(
        run_id=report["evaluation_run_id"],
        report_sha256=digest,
        clean_raw={k: float(raw[k]) for k in METRIC_KEYS},
        clean_dedup_raw={k: float(dedup[k]) for k in METRIC_KEYS},
        test_predictions=predictions,
        test_predictions_sha256=test_sha,
    )


def evaluate_robustness(
    config: RobustnessConfig,
    *,
    dataset_spec: DatasetSpec = SPEECH_COMMANDS_KW10,
    root: Path | None = None,
    run_id: str | None = None,
    allow_dirty_source: bool = False,
    corruptions: tuple[CorruptionSpec, ...] = CORRUPTIONS,
) -> RobustnessResult:
    """Run the protocol; see the module docstring. ``corruptions`` exists for tests only."""
    loaded = verify_checkpoint(config.checkpoint, config.data)
    window = WindowSection.model_validate(loaded.metadata["preprocessing"]["window"])
    manifest = check_manifest_inputs(config.data, window, dataset_spec)
    baseline = _check_baseline(config, loaded, manifest)
    backend = codec_backend_info() if any(c.name == "codec" for c in corruptions) else {}
    source = current_source_revision()
    if source.dirty and not allow_dirty_source:
        raise ProvenanceError(
            f"working tree has uncommitted changes at {source.commit}; commit them "
            "(or pass allow_dirty_source for a non-result development run)"
        )
    run = start_run(config, root=root, run_id=run_id)
    try:
        report, summary = _run(
            config, run, loaded, manifest, window, baseline, backend, corruptions
        )
    except BaseException as exc:
        _write_atomic(
            run.path / "summary.json",
            _json_bytes(
                {"status": "failed", "run_id": run.run_id, "error": f"{type(exc).__name__}: {exc}"}
            ),
        )
        raise
    return RobustnessResult(run=run, report=report, summary=summary)


def _float_windows(clips: list[npt.NDArray[np.float32]], num_samples: int) -> torch.Tensor:
    windows = np.zeros((len(clips), num_samples), dtype=np.float32)
    for i, clip in enumerate(clips):
        if clip.size > num_samples:
            raise RobustnessError(f"corrupted clip {i} longer than the window")
        windows[i, : clip.size] = clip
    return torch.from_numpy(windows)


def _manifest_csv(
    rows: list[ManifestRow], spec: CorruptionSpec, severity: Severity, clips: list[CorruptedClip]
) -> bytes:
    measure_names = sorted(clips[0].measures) if clips else []
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "relative_path",
            "source_audio_sha256",
            "corruption",
            "version",
            "severity",
            "seed",
            "output_sha256",
            *measure_names,
        ]
    )
    for row, clip in zip(rows, clips, strict=True):
        if sorted(clip.measures) != measure_names:
            raise RobustnessError("corruption measures differ between examples")
        writer.writerow(
            [
                row.relative_path,
                row.audio_sha256,
                spec.name,
                spec.version,
                severity.name,
                clip.seed,
                output_sha256(clip.samples),
                *(repr(clip.measures[m]) for m in measure_names),
            ]
        )
    return buffer.getvalue().encode("utf-8")


def _summarise_measures(clips: list[CorruptedClip]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in sorted(clips[0].measures):
        values = np.array([c.measures[name] for c in clips], dtype=np.float64)
        out[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def _headline(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(metrics[k]) for k in (*METRIC_KEYS, "examples")}


def _run(
    config: RobustnessConfig,
    run: RunDirectory,
    loaded: LoadedCheckpoint,
    manifest: ValidatedManifest,
    window: WindowSection,
    baseline: Baseline,
    backend: dict[str, str],
    corruptions: tuple[CorruptionSpec, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    options = config.robustness
    if options.num_threads is not None:
        torch.set_num_threads(options.num_threads)
    labels = manifest.labels
    rows = manifest.rows_for(SPLIT)
    audio = read_split_audio(manifest, SPLIT, config.data, window)
    clean_clips = [audio.waveforms[i, : row.num_frames].numpy() for i, row in enumerate(rows)]
    y_true = audio.labels.numpy().astype(np.int64)

    def predictions_for(logits: npt.NDArray[np.float32]) -> PredictionSet:
        return PredictionSet(
            split=SPLIT,
            labels=labels,
            relative_paths=audio.relative_paths,
            audio_sha256=tuple(r.audio_sha256 for r in rows),
            speaker_ids=tuple(r.speaker_id for r in rows),
            y_true=y_true,
            logits=logits,
        )

    def infer(clips: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        return infer_logits(
            loaded.model,
            loaded.frontend,
            _float_windows(clips, window.num_samples),
            batch_size=options.batch_size,
            num_classes=len(labels),
        )

    # Clean-path reproduction of the Milestone 4 raw test logits.
    if baseline.test_predictions.relative_paths != audio.relative_paths:
        raise RobustnessError("baseline test predictions are not in manifest order")
    clean_float = [(c.astype(np.float64) / 32768.0).astype(np.float32) for c in clean_clips]
    clean_logits = infer(clean_float)
    if not np.array_equal(clean_logits, baseline.test_predictions.logits):
        worst = float(np.max(np.abs(clean_logits - baseline.test_predictions.logits)))
        raise RobustnessError(
            f"robustness path does not reproduce the Milestone 4 test logits (max |diff| {worst})"
        )
    clean_check = {"reproduced_baseline_test_logits": True, "examples": len(clean_clips)}

    artifacts: dict[str, str] = {}
    conditions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=options.corruption_workers) as pool:
        for spec in corruptions:
            for severity in spec.severities:
                corrupted = list(
                    pool.map(
                        lambda i, s=spec, v=severity: corrupt(
                            s, v, clean_clips[i], rows[i].relative_path
                        ),
                        range(len(rows)),
                    )
                )
                condition_dir = f"conditions/{spec.name}/{severity.name}"
                logits = infer([c.samples for c in corrupted])
                predictions = predictions_for(logits)
                pred_path = run.path / condition_dir / "predictions.csv"
                artifacts[f"{condition_dir}/predictions.csv"] = _write_atomic(
                    pred_path, encode_predictions(predictions)
                )
                reread = decode_predictions(pred_path.read_bytes(), split=SPLIT, labels=labels)
                artifacts[f"{condition_dir}/corruption_manifest.csv"] = _write_atomic(
                    run.path / condition_dir / "corruption_manifest.csv",
                    _manifest_csv(rows, spec, severity, corrupted),
                )
                for index in range(min(options.retain_audio_examples, len(rows))):
                    name = rows[index].relative_path.replace("/", "__")
                    wav_path = run.path / "audio" / spec.name / severity.name / name
                    _write_atomic(wav_path, encode_wav_float32(corrupted[index].samples, 16000))
                    check = parse_wav(wav_path.read_bytes())
                    if check.sample_rate_hz != 16000 or not np.array_equal(
                        check.samples_float32(), corrupted[index].samples
                    ):
                        raise RobustnessError(f"retained audio {wav_path} does not read back")
                metrics = classification_metrics(
                    reread.logits, reread.y_true, labels, n_bins=ECE_BINS
                )
                kept = reread.subset(_dedup_indices(reread))
                dedup = classification_metrics(kept.logits, kept.y_true, labels, n_bins=ECE_BINS)
                conditions.append(
                    {
                        "corruption": spec.name,
                        "version": spec.version,
                        "severity": severity.name,
                        "params": dict(severity.params),
                        "metrics": metrics,
                        "degradation": degradation(baseline.clean_raw, metrics),
                        "top_confusions": top_confusions(metrics),
                        "measures": _summarise_measures(corrupted),
                        "duplicate_sensitivity": {
                            "rows_kept": len(kept),
                            "metrics": _headline(dedup),
                            "degradation": degradation(baseline.clean_dedup_raw, dedup),
                        },
                    }
                )

    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "protocol": {
            "id": PROTOCOL_ID,
            "seed": PROTOCOL_SEED,
            "seed_scheme": "derive_seed(protocol_seed, protocol_id, corruption, version, "
            "severity, relative_path) -> numpy.random.default_rng",
            "split": SPLIT,
            "probabilities": "raw softmax (T = 1); no calibration refit",
            "degradation": "clean metric - corrupted metric (Milestone 4 official raw test)",
            "ece_bins": ECE_BINS,
            "corruptions": [spec.describe() for spec in corruptions],
        },
        "codec_backend": backend,
        "numpy_version": np.__version__,
        "robustness_run_id": run.run_id,
        "source": run.metadata["source"],
        "checkpoint": {
            "path": config.checkpoint.path.as_posix(),
            "sha256": loaded.sha256,
            "run_id": loaded.metadata["run_id"],
            "epoch": loaded.metadata["epoch"],
        },
        "manifest": {"sha256": manifest.sha256, "test_rows": len(rows)},
        "labels": list(labels),
        "baseline": {
            "evaluation_run_id": baseline.run_id,
            "report_sha256": baseline.report_sha256,
            "test_predictions_sha256": baseline.test_predictions_sha256,
            "clean_raw": baseline.clean_raw,
            "clean_dedup_raw": baseline.clean_dedup_raw,
        },
        "clean_path_check": clean_check,
        "conditions": conditions,
        "artifacts_sha256": artifacts,
        "batch_size": options.batch_size,
        "torch_num_threads": torch.get_num_threads(),
    }
    all_artifacts = dict(artifacts)
    all_artifacts["report.json"] = _write_atomic(run.path / "report.json", _json_bytes(report))
    all_artifacts["report.md"] = _write_atomic(run.path / "report.md", _markdown(report))
    summary = {
        "status": "completed",
        "run_id": run.run_id,
        "protocol": PROTOCOL_ID,
        "conditions": len(conditions),
        "clean_raw": baseline.clean_raw,
        "headline": [
            {
                "corruption": c["corruption"],
                "severity": c["severity"],
                **{k: c["metrics"][k] for k in METRIC_KEYS},
                "degradation": c["degradation"],
            }
            for c in conditions
        ],
        "artifacts_sha256": all_artifacts,
        "seconds": round(time.perf_counter() - started, 3),
    }
    _write_atomic(run.path / "summary.json", _json_bytes(summary))
    return report, summary


def _markdown(report: dict[str, Any]) -> bytes:
    clean = report["baseline"]["clean_raw"]
    lines = [
        f"# Robustness report: {report['protocol']['id']}",
        "",
        f"- Run `{report['robustness_run_id']}` at `{report['source']['commit']}` "
        f"(dirty {report['source']['dirty']})",
        f"- Checkpoint `{report['checkpoint']['sha256']}` (run `{report['checkpoint']['run_id']}`, "
        f"epoch {report['checkpoint']['epoch']}); manifest `{report['manifest']['sha256']}`",
        f"- Clean baseline: Milestone 4 run `{report['baseline']['evaluation_run_id']}` "
        f"(report SHA-256 `{report['baseline']['report_sha256']}`), raw test: accuracy "
        f"{clean['accuracy']:.4f}, macro F1 {clean['macro_f1']:.4f}, NLL {clean['nll']:.4f}, "
        f"ECE {clean['ece']:.4f}",
        f"- Clean-path check: the Milestone 4 test logits were reproduced bit-for-bit: "
        f"{report['clean_path_check']['reproduced_baseline_test_logits']}",
        f"- Codec backend: {report['codec_backend'] or 'not used'}",
        "",
        "Raw probabilities, all 4,074 test rows (duplicates retained). Degradation = clean - "
        "corrupted: a positive value is worse for accuracy/F1, a negative value is worse "
        "for NLL/ECE.",
        "",
        "| Corruption | Severity | Accuracy | Δ acc | Macro F1 | Δ F1 | NLL | Δ NLL | ECE "
        "| Δ ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in report["conditions"]:
        m, d = c["metrics"], c["degradation"]
        lines.append(
            f"| {c['corruption']} | {c['severity']} | {m['accuracy']:.4f} | {d['accuracy']:+.4f} | "
            f"{m['macro_f1']:.4f} | {d['macro_f1']:+.4f} | {m['nll']:.4f} | {d['nll']:+.4f} | "
            f"{m['ece']:.4f} | {d['ece']:+.4f} |"
        )
    lines += ["", "## Largest confusions per condition (true → predicted: count)", ""]
    for c in report["conditions"]:
        text = ", ".join(
            f"{t['true']} → {t['predicted']}: {t['count']}" for t in c["top_confusions"][:3]
        )
        lines.append(f"- {c['corruption']} / {c['severity']}: {text}")
    lines += [
        "",
        "## Secondary: one row per audio_sha256 (not the official result)",
        "",
        "| Corruption | Severity | Rows | Accuracy | Δ acc (vs dedup clean) | Macro F1 "
        "| NLL | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in report["conditions"]:
        s = c["duplicate_sensitivity"]
        m = s["metrics"]
        lines.append(
            f"| {c['corruption']} | {c['severity']} | {s['rows_kept']} | {m['accuracy']:.4f} | "
            f"{s['degradation']['accuracy']:+.4f} | {m['macro_f1']:.4f} | {m['nll']:.4f} | "
            f"{m['ece']:.4f} |"
        )
    lines += [
        "",
        "Full precision, confusion matrices, measures, and artifact hashes: `report.json`.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")
