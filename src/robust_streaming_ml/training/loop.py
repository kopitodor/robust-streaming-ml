"""Baseline training entry point.

:func:`train` is the only supported way to train. In order it:

1. checks that the config, validated manifest, sidecar, and validation report
   are compatible (:func:`~robust_streaming_ml.training.data.load_validated_manifest`),
   with no side effects;
2. resolves the source revision, refusing a dirty working tree unless
   ``allow_dirty_source`` is set;
3. calls :func:`~robust_streaming_ml.run_dir.start_run`, which records
   provenance, applies seeding, and creates the run directory;
4. loads and re-verifies the ``train`` and ``validation`` audio (``test`` is
   never loaded);
5. trains on ``train`` only. After each epoch it computes validation loss and
   accuracy, keeps the checkpoint with the lowest validation loss as ``best``,
   and optionally writes ``last``.

Run-directory contents: ``config.yaml`` and ``metadata.json`` (from
``start_run``), ``train_log.jsonl`` (one JSON object per epoch),
``checkpoints/{best,last}.{pt,json}``, and ``summary.json``. If training
fails after the run directory exists, ``summary.json`` records
``status: failed`` with the error, and the exception propagates.

Validation loss and accuracy are monitoring values for checkpoint selection.
The formal evaluation (held-out test, macro F1, calibration) is Milestone 4.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch
import torch.nn.functional as F

from robust_streaming_ml.checkpoint import save_checkpoint
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import DatasetSpec
from robust_streaming_ml.features import (
    FEATURE_VERSION,
    LogMelSpectrogram,
    feature_fingerprint,
    pcm16_to_float,
)
from robust_streaming_ml.model import KeywordCNN, count_parameters, layer_shapes
from robust_streaming_ml.run_dir import (
    ProvenanceError,
    RunDirectory,
    config_digest,
    current_source_revision,
    start_run,
)
from robust_streaming_ml.seed import derive_seed
from robust_streaming_ml.training.config import TrainingConfig
from robust_streaming_ml.training.data import (
    EpochSampler,
    ValidatedManifest,
    WaveformDataset,
    load_split_audio,
    load_validated_manifest,
    make_loader,
)

__all__ = ["LOG_FILENAME", "SUMMARY_FILENAME", "TrainingError", "TrainingResult", "train"]

LOG_FILENAME = "train_log.jsonl"
SUMMARY_FILENAME = "summary.json"
CHECKPOINT_DIR = "checkpoints"


class TrainingError(RuntimeError):
    """Raised when training cannot continue (for example a non-finite loss)."""


@dataclass(frozen=True)
class TrainingResult:
    run: RunDirectory
    summary: dict[str, Any]

    @property
    def best_checkpoint(self) -> Path:
        return self.run.path / CHECKPOINT_DIR / "best.pt"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate(
    model: KeywordCNN, frontend: LogMelSpectrogram, loader: Any
) -> tuple[float, float, int]:
    """Mean cross-entropy and accuracy over a loader (eval mode, no gradients)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for waveforms, labels in loader:
            logits = model(frontend(pcm16_to_float(waveforms)))
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += int(labels.shape[0])
    return total_loss / count, correct / count, count


def train(
    config: TrainingConfig,
    *,
    dataset_spec: DatasetSpec = SPEECH_COMMANDS_KW10,
    root: Path | None = None,
    run_id: str | None = None,
    allow_dirty_source: bool = False,
) -> TrainingResult:
    """Train the baseline; see the module docstring for the exact procedure."""
    manifest = load_validated_manifest(config, dataset_spec)
    source = current_source_revision()
    if source.dirty and not allow_dirty_source:
        raise ProvenanceError(
            f"working tree has uncommitted changes at {source.commit}; commit them "
            "(or pass allow_dirty_source for a non-result development run)"
        )
    run = start_run(config, root=root, run_id=run_id)
    try:
        summary = _train(config, run, manifest)
    except BaseException as exc:
        _write_json(
            run.path / SUMMARY_FILENAME,
            {"status": "failed", "run_id": run.run_id, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    return TrainingResult(run=run, summary=summary)


def _train(
    config: TrainingConfig, run: RunDirectory, manifest: ValidatedManifest
) -> dict[str, Any]:
    cfg = config.training
    if cfg.num_threads is not None:
        torch.set_num_threads(cfg.num_threads)
    seed = config.seed
    derived = {
        "model_init": derive_seed(seed, "model_init"),
        "dropout": derive_seed(seed, "dropout"),
        "loader": derive_seed(seed, "loader"),
        "shuffle": "derive_seed(seed, 'shuffle', epoch)",
        "augment": "derive_seed(seed, 'augment', epoch, index)",
    }

    started = time.perf_counter()
    train_audio = load_split_audio(manifest, "train", config)
    val_audio = load_split_audio(manifest, "validation", config)
    load_seconds = time.perf_counter() - started

    frontend = LogMelSpectrogram(config.features, config.window)
    input_shape = frontend.output_shape
    torch.manual_seed(derived["model_init"])
    model = KeywordCNN(
        config.model, num_classes=len(manifest.labels), n_mels=config.features.n_mels
    )
    num_parameters = count_parameters(model)
    shapes = layer_shapes(model, input_shape)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.lr_schedule == "cosine"
        else None
    )
    torch.manual_seed(derived["dropout"])

    train_sampler = EpochSampler(len(train_audio), seed=seed, shuffle=True)
    val_sampler = EpochSampler(len(val_audio), seed=seed, shuffle=False)
    sample_rate = config.window.sample_rate_hz
    train_loader = make_loader(
        WaveformDataset(
            train_audio, seed=seed, sample_rate_hz=sample_rate, augmentation=cfg.augmentation
        ),
        train_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=seed,
    )
    val_loader = make_loader(
        WaveformDataset(val_audio, seed=seed, sample_rate_hz=sample_rate),
        val_sampler,
        batch_size=cfg.validation_batch_size,
        num_workers=cfg.num_workers,
        seed=seed,
    )

    fingerprint = feature_fingerprint(config.features, config.window)
    base_metadata: dict[str, Any] = {
        "run_id": run.run_id,
        "labels": list(manifest.labels),
        "config": config.to_dict(),
        "config_sha256": config_digest(config),
        "model": {
            "architecture": config.model.architecture,
            "config": config.model.model_dump(mode="json"),
            "num_classes": len(manifest.labels),
            "num_parameters": num_parameters,
            "input_shape": list(input_shape),
            "layer_shapes": shapes,
        },
        "preprocessing": {
            "window": config.window.model_dump(mode="json"),
            "features": config.features.model_dump(mode="json"),
            "feature_version": FEATURE_VERSION,
            "feature_fingerprint": fingerprint,
            "input_scaling": "int16 / 32768",
        },
        "manifest": {
            "dataset": manifest.dataset,
            "sha256": manifest.sha256,
            "path": manifest.path.as_posix(),
            "rows_by_split": manifest.rows_by_split,
            "splits_used": ["train", "validation"],
        },
        "source": run.metadata["source"],
        "seeding": {"applied": run.metadata["seeding"], "derived": derived},
        "torch_version": torch.__version__,
    }

    checkpoint_dir = run.path / CHECKPOINT_DIR
    best: dict[str, Any] = {"validation_loss": math.inf}
    last_checkpoint: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    log: TextIO
    with (run.path / LOG_FILENAME).open("w", encoding="utf-8") as log:
        for epoch in range(1, cfg.epochs + 1):
            epoch_started = time.perf_counter()
            train_sampler.set_epoch(epoch)
            model.train()
            learning_rate = optimizer.param_groups[0]["lr"]
            total_loss, correct, count = 0.0, 0, 0
            for waveforms, labels in train_loader:
                with torch.no_grad():
                    features = frontend(pcm16_to_float(waveforms))
                logits = model(features)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise TrainingError(f"non-finite training loss in epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                batch = int(labels.shape[0])
                total_loss += float(loss.detach()) * batch
                correct += int((logits.detach().argmax(dim=1) == labels).sum())
                count += batch
            if scheduler is not None:
                scheduler.step()
            val_loss, val_accuracy, val_count = _evaluate(model, frontend, val_loader)
            if not math.isfinite(val_loss):
                raise TrainingError(f"non-finite validation loss in epoch {epoch}")
            improved = val_loss < best["validation_loss"]
            record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": total_loss / count,
                "train_accuracy": correct / count,
                "train_examples": count,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "validation_examples": val_count,
                "is_best": improved,
                "seconds": round(time.perf_counter() - epoch_started, 3),
            }
            validation = {"loss": val_loss, "accuracy": val_accuracy, "epoch": epoch}
            if improved:
                sha = save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    {**base_metadata, "kind": "best", "epoch": epoch, "validation": validation},
                )
                best = {
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                    "file": "best.pt",
                    "sha256": sha,
                }
            if config.checkpoint.save_last:
                sha = save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model,
                    {**base_metadata, "kind": "last", "epoch": epoch, "validation": validation},
                )
                last_checkpoint = {"epoch": epoch, "file": "last.pt", "sha256": sha}
            history.append(record)
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()

    summary: dict[str, Any] = {
        "status": "completed",
        "run_id": run.run_id,
        "epochs_completed": len(history),
        "best": best,
        "last": last_checkpoint,
        "final_epoch": history[-1],
        "selection_metric": cfg.selection_metric,
        "num_parameters": num_parameters,
        "layer_shapes": shapes,
        "rows_loaded": {"train": len(train_audio), "validation": len(val_audio), "test": 0},
        "held_out_test": "not loaded or evaluated (Milestone 4)",
        "manifest_sha256": manifest.sha256,
        "feature_fingerprint": fingerprint,
        "derived_seeds": derived,
        "torch_num_threads": torch.get_num_threads(),
        "audio_load_seconds": round(load_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(run.path / SUMMARY_FILENAME, summary)
    return summary
