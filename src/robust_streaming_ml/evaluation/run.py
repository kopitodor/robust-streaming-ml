"""Formal evaluation entry point (protocol ``kw10-clean-eval-v1``).

:func:`evaluate` checks every gate before writing anything
(docs/evaluation-protocol.md, section 2), starts a run with ``start_run``,
and then follows the protocol's order of operations:

1. predict on validation and write ``predictions/validation.csv``;
2. fit the temperature on the predictions read back from that file and write
   ``calibration.json``;
3. predict on test and write ``predictions/test.csv``;
4. write the calibrated probability files, then compute every metric from the
   artifacts and write ``report.json``, ``report.md``, and ``summary.json``.

All files are written atomically. If anything fails after the run directory
exists, ``summary.json`` records ``status: failed``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from robust_streaming_ml.checkpoint import CheckpointError, LoadedCheckpoint, load_checkpoint
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import DatasetSpec
from robust_streaming_ml.evaluation.calibration import (
    calibration_record,
    fit_temperature,
    load_calibration,
    save_calibration,
)
from robust_streaming_ml.evaluation.config import (
    ECE_BINS,
    PROTOCOL,
    CheckpointRef,
    EvaluationConfig,
)
from robust_streaming_ml.evaluation.metrics import classification_metrics
from robust_streaming_ml.evaluation.predictions import (
    PredictionSet,
    decode_predictions,
    encode_calibrated,
    encode_predictions,
    predict_split,
)
from robust_streaming_ml.run_dir import (
    METADATA_FILENAME,
    ProvenanceError,
    RunDirectory,
    commit_is_ancestor_of_head,
    current_source_revision,
    start_run,
)
from robust_streaming_ml.training.config import DataSection, WindowSection
from robust_streaming_ml.training.data import (
    ValidatedManifest,
    check_manifest_inputs,
    read_split_audio,
)

__all__ = [
    "EVALUATION_SPLITS",
    "EvaluationError",
    "EvaluationResult",
    "evaluate",
    "verify_checkpoint",
]

EVALUATION_SPLITS = ("validation", "test")
REPORT_SCHEMA_VERSION = 1


class EvaluationError(RuntimeError):
    """Raised when a gate fails or evaluation cannot be completed."""


@dataclass(frozen=True)
class EvaluationResult:
    run: RunDirectory
    report: dict[str, Any]
    summary: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return _sha256_bytes(data)


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_checkpoint(checkpoint: CheckpointRef, data: DataSection) -> LoadedCheckpoint:
    """Gates 1-3: pinned hash, compatibility, and checkpoint provenance (no side effects)."""
    path = checkpoint.path
    if not path.is_file():
        raise EvaluationError(f"checkpoint not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != checkpoint.sha256:
        raise EvaluationError(f"checkpoint SHA-256 {digest} != pinned {checkpoint.sha256}")
    try:
        loaded = load_checkpoint(
            path,
            expect_manifest_sha256=data.manifest_sha256,
            expect_labels=data.labels,
        )
    except CheckpointError as exc:
        raise EvaluationError(f"incompatible checkpoint: {exc}") from exc
    meta = loaded.metadata
    if meta["manifest"]["dataset"] != data.dataset:
        raise EvaluationError("checkpoint was trained on another dataset")
    if meta["manifest"].get("splits_used") != ["train", "validation"]:
        raise EvaluationError("checkpoint does not record training on train/validation only")
    source = meta["source"]
    if source["dirty"]:
        raise EvaluationError("checkpoint was trained from a dirty working tree")
    if not commit_is_ancestor_of_head(source["commit"]):
        raise EvaluationError(
            f"checkpoint commit {source['commit']} is not an ancestor of the evaluation commit"
        )
    run_metadata_path = path.parent.parent / METADATA_FILENAME
    try:
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"training run metadata unreadable: {run_metadata_path}") from exc
    for key in ("run_id", "source", "config_sha256"):
        if run_metadata.get(key) != meta[key]:
            raise EvaluationError(f"training run metadata {key} does not match the checkpoint")
    return loaded


def evaluate(
    config: EvaluationConfig,
    *,
    dataset_spec: DatasetSpec = SPEECH_COMMANDS_KW10,
    root: Path | None = None,
    run_id: str | None = None,
    allow_dirty_source: bool = False,
) -> EvaluationResult:
    """Run the protocol; see the module docstring."""
    loaded = verify_checkpoint(config.checkpoint, config.data)
    window = WindowSection.model_validate(loaded.metadata["preprocessing"]["window"])
    manifest = check_manifest_inputs(config.data, window, dataset_spec)
    source = current_source_revision()
    if source.dirty and not allow_dirty_source:
        raise ProvenanceError(
            f"working tree has uncommitted changes at {source.commit}; commit them "
            "(or pass allow_dirty_source for a non-result development run)"
        )
    run = start_run(config, root=root, run_id=run_id)
    try:
        report, summary = _evaluate(config, run, loaded, manifest, window)
    except BaseException as exc:
        _write_atomic(
            run.path / "summary.json",
            _json_bytes(
                {"status": "failed", "run_id": run.run_id, "error": f"{type(exc).__name__}: {exc}"}
            ),
        )
        raise
    return EvaluationResult(run=run, report=report, summary=summary)


def _dedup_indices(predictions: PredictionSet) -> list[int]:
    """One row per ``audio_sha256``: the one with the smallest ``relative_path``."""
    chosen: dict[str, int] = {}
    for i, (digest, path) in enumerate(
        zip(predictions.audio_sha256, predictions.relative_paths, strict=True)
    ):
        if digest not in chosen or path < predictions.relative_paths[chosen[digest]]:
            chosen[digest] = i
    return sorted(chosen.values())


def _predict(
    config: EvaluationConfig,
    loaded: LoadedCheckpoint,
    manifest: ValidatedManifest,
    window: WindowSection,
    split: str,
) -> PredictionSet:
    if split not in EVALUATION_SPLITS:
        raise EvaluationError(f"split {split!r} is not evaluated by this protocol")
    audio = read_split_audio(manifest, split, config.data, window)
    return predict_split(
        loaded.model,
        loaded.frontend,
        audio,
        manifest.rows_for(split),
        manifest.labels,
        batch_size=config.evaluation.batch_size,
    )


def _evaluate(
    config: EvaluationConfig,
    run: RunDirectory,
    loaded: LoadedCheckpoint,
    manifest: ValidatedManifest,
    window: WindowSection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    if config.evaluation.num_threads is not None:
        torch.set_num_threads(config.evaluation.num_threads)
    labels = manifest.labels
    meta = loaded.metadata
    pred_dir = run.path / "predictions"
    artifacts: dict[str, str] = {}

    # 1. validation predictions
    val_path = pred_dir / "validation.csv"
    artifacts["predictions/validation.csv"] = _write_atomic(
        val_path, encode_predictions(_predict(config, loaded, manifest, window, "validation"))
    )
    validation = decode_predictions(val_path.read_bytes(), split="validation", labels=labels)

    # 2. calibration, fitted on validation only, persisted before test inference
    fit = fit_temperature(validation)
    compatibility = {
        "checkpoint_sha256": loaded.sha256,
        "checkpoint_run_id": meta["run_id"],
        "checkpoint_epoch": meta["epoch"],
        "labels": list(labels),
        "feature_fingerprint": meta["preprocessing"]["feature_fingerprint"],
        "manifest_sha256": manifest.sha256,
        "protocol": PROTOCOL["id"],
    }
    record = calibration_record(
        fit,
        fitted_on={
            "examples": len(validation),
            "predictions_file": "predictions/validation.csv",
            "predictions_sha256": artifacts["predictions/validation.csv"],
            "manifest_sha256": manifest.sha256,
        },
        compatibility=compatibility,
    )
    calibration_path = run.path / "calibration.json"
    save_calibration(calibration_path, record)
    artifacts["calibration.json"] = _sha256_bytes(calibration_path.read_bytes())
    reloaded = load_calibration(
        calibration_path, expect_checkpoint_sha256=loaded.sha256, expect_labels=labels
    )
    temperature = float(reloaded["parameters"]["temperature"])

    # 3. test predictions (only after calibration is fixed)
    test_path = pred_dir / "test.csv"
    artifacts["predictions/test.csv"] = _write_atomic(
        test_path, encode_predictions(_predict(config, loaded, manifest, window, "test"))
    )
    test = decode_predictions(test_path.read_bytes(), split="test", labels=labels)

    # 4. calibrated probabilities and metrics, all from the artifacts
    splits = {"validation": validation, "test": test}
    for name, predictions in splits.items():
        artifacts[f"predictions/{name}_calibrated.csv"] = _write_atomic(
            pred_dir / f"{name}_calibrated.csv", encode_calibrated(predictions, temperature)
        )
    expected_rows = manifest.rows_by_split
    metrics: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    bins = ECE_BINS
    for name, predictions in splits.items():
        if len(predictions) != expected_rows[name]:
            raise EvaluationError(
                f"{name}: {len(predictions)} predictions for {expected_rows[name]} rows"
            )
        metrics[name] = {
            "raw": classification_metrics(
                predictions.logits, predictions.y_true, labels, n_bins=bins
            ),
            "calibrated": classification_metrics(
                predictions.logits, predictions.y_true, labels, temperature=temperature, n_bins=bins
            ),
        }
        metrics[name]["calibrated"]["in_sample"] = name == "validation"
        kept = _dedup_indices(predictions)
        subset = predictions.subset(kept)
        sensitivity[name] = {
            "rows_official": len(predictions),
            "rows_kept": len(kept),
            "rows_removed": len(predictions) - len(kept),
            "raw": classification_metrics(subset.logits, subset.y_true, labels, n_bins=bins),
            "calibrated": classification_metrics(
                subset.logits, subset.y_true, labels, temperature=temperature, n_bins=bins
            ),
        }

    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "evaluation_run_id": run.run_id,
        "evaluation_source": run.metadata["source"],
        "checkpoint": {
            "path": config.checkpoint.path.as_posix(),
            "sha256": loaded.sha256,
            "run_id": meta["run_id"],
            "kind": meta["kind"],
            "epoch": meta["epoch"],
            "source": meta["source"],
            "config_sha256": meta["config_sha256"],
            "num_parameters": meta["model"]["num_parameters"],
            "feature_fingerprint": meta["preprocessing"]["feature_fingerprint"],
            "training_validation_monitor": meta["validation"],
        },
        "manifest": {
            "dataset": manifest.dataset,
            "sha256": manifest.sha256,
            "rows_by_split": expected_rows,
        },
        "labels": list(labels),
        "calibration": {
            "method": "temperature_scaling",
            **fit.as_dict(),
            "fitted_on": "validation",
        },
        "metrics": metrics,
        "duplicate_sensitivity": {
            "note": "secondary view; the official result is 'metrics'",
            **sensitivity,
        },
        "artifacts_sha256": artifacts,
        "batch_size": config.evaluation.batch_size,
        "torch_num_threads": torch.get_num_threads(),
    }
    report_bytes = _json_bytes(report)
    artifacts_all = dict(artifacts)
    artifacts_all["report.json"] = _write_atomic(run.path / "report.json", report_bytes)
    artifacts_all["report.md"] = _write_atomic(run.path / "report.md", _markdown(report))
    summary = {
        "status": "completed",
        "run_id": run.run_id,
        "protocol": PROTOCOL["id"],
        "checkpoint_sha256": loaded.sha256,
        "temperature": temperature,
        "headline": {
            split: {
                variant: {
                    k: metrics[split][variant][k]
                    for k in ("examples", "accuracy", "macro_f1", "nll", "ece")
                }
                for variant in ("raw", "calibrated")
            }
            for split in splits
        },
        "artifacts_sha256": artifacts_all,
        "seconds": round(time.perf_counter() - started, 3),
    }
    _write_atomic(run.path / "summary.json", _json_bytes(summary))
    return report, summary


def _markdown(report: dict[str, Any]) -> bytes:
    labels = report["labels"]
    ckpt = report["checkpoint"]
    cal = report["calibration"]
    lines = [
        f"# Evaluation report: {report['protocol']['id']}",
        "",
        f"- Evaluation run: `{report['evaluation_run_id']}` "
        f"(source `{report['evaluation_source']['commit']}`, dirty "
        f"{report['evaluation_source']['dirty']})",
        f"- Checkpoint: `{ckpt['path']}` (SHA-256 `{ckpt['sha256']}`), run `{ckpt['run_id']}`, "
        f"epoch {ckpt['epoch']}, trained at `{ckpt['source']['commit']}`",
        f"- Manifest: `{report['manifest']['sha256']}`",
        "- Calibration: temperature scaling fitted on validation only: "
        f"T = {cal['temperature']:.6f} (validation NLL {cal['nll_before']:.6f} -> "
        f"{cal['nll_after']:.6f}; at bound: {cal['at_bound']})",
        "",
        "## Official results (all rows, duplicates retained)",
        "",
        "| Split | Probabilities | N | Accuracy | Macro F1 | NLL | ECE (15 bins) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("validation", "test"):
        for variant in ("raw", "calibrated"):
            m = report["metrics"][split][variant]
            tag = variant + (" (in-sample)" if m.get("in_sample") else "")
            lines.append(
                f"| {split} | {tag} | {m['examples']} | {m['accuracy']:.4f} | "
                f"{m['macro_f1']:.4f} | {m['nll']:.4f} | {m['ece']:.4f} |"
            )
    test_raw = report["metrics"]["test"]["raw"]
    lines += ["", "## Test per-class (raw)", "", "| Label | Precision | Recall | F1 | Support |"]
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label in labels:
        pc = test_raw["per_class"][label]
        lines.append(
            f"| {label} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | "
            f"{pc['support']} |"
        )
    lines += ["", "## Test confusion matrix (rows = true, columns = predicted)", ""]
    lines.append("| true \\ pred | " + " | ".join(labels) + " |")
    lines.append("| --- |" + " ---: |" * len(labels))
    for label, row in zip(labels, test_raw["confusion_matrix"]["rows_true_cols_pred"], strict=True):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        "## Secondary: duplicate-content sensitivity (not the official result)",
        "",
        "| Split | Probabilities | Rows kept (removed) | Accuracy | Macro F1 | NLL | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("validation", "test"):
        s = report["duplicate_sensitivity"][split]
        for variant in ("raw", "calibrated"):
            m = s[variant]
            lines.append(
                f"| {split} | {variant} | {s['rows_kept']} ({s['rows_removed']}) | "
                f"{m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['nll']:.4f} | {m['ece']:.4f} |"
            )
    lines += [
        "",
        "Full-precision values, per-class metrics, ECE bins, and artifact hashes are in "
        "`report.json`.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")
