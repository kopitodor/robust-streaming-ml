"""Deterministic inference and regenerable prediction artifacts.

``predictions/<split>.csv`` holds one row per manifest row, in manifest order,
with these columns:
- ``relative_path, audio_sha256, speaker_id, label, label_index``;
- ``pred_index, pred_label, correct``;
- ``logit_<label>`` for every label: float32 values written with a repr that
  round-trips exactly;
- ``prob_<label>`` for every label: ``softmax(logits)`` in float64.

``predictions/<split>_calibrated.csv`` adds, per row,
``relative_path, pred_index, cal_prob_<label>...``, computed from the stored
logits and the fitted temperature.

:func:`decode_predictions` re-derives the argmax, correctness, label index,
and raw probabilities from the stored logits and rejects any row that
disagrees, so every aggregate can be regenerated from the artifact alone.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from robust_streaming_ml.data.manifest import SPLITS, ManifestRow
from robust_streaming_ml.evaluation.metrics import predictions_from_logits, softmax
from robust_streaming_ml.features import LogMelSpectrogram, pcm16_to_float
from robust_streaming_ml.model import KeywordCNN
from robust_streaming_ml.training.data import SplitAudio

__all__ = [
    "PredictionError",
    "PredictionSet",
    "decode_predictions",
    "encode_calibrated",
    "encode_predictions",
    "infer_logits",
    "predict_split",
    "prediction_columns",
]


class PredictionError(ValueError):
    """Raised when predictions or a prediction artifact are inconsistent."""


@dataclass(frozen=True)
class PredictionSet:
    split: str
    labels: tuple[str, ...]
    relative_paths: tuple[str, ...]
    audio_sha256: tuple[str, ...]
    speaker_ids: tuple[str, ...]
    y_true: npt.NDArray[np.int64]
    logits: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        n = len(self.relative_paths)
        if self.split not in SPLITS:
            raise PredictionError(f"unknown split {self.split!r}")
        if not (len(self.audio_sha256) == len(self.speaker_ids) == self.y_true.shape[0] == n):
            raise PredictionError("prediction fields have different lengths")
        if self.logits.shape != (n, len(self.labels)) or self.logits.dtype != np.float32:
            raise PredictionError(f"logits must be float32 [{n}, {len(self.labels)}]")

    def __len__(self) -> int:
        return len(self.relative_paths)

    @property
    def y_pred(self) -> npt.NDArray[np.int64]:
        return predictions_from_logits(self.logits)

    def subset(self, indices: list[int]) -> PredictionSet:
        idx = np.asarray(indices, dtype=np.int64)
        return PredictionSet(
            split=self.split,
            labels=self.labels,
            relative_paths=tuple(self.relative_paths[i] for i in indices),
            audio_sha256=tuple(self.audio_sha256[i] for i in indices),
            speaker_ids=tuple(self.speaker_ids[i] for i in indices),
            y_true=self.y_true[idx],
            logits=self.logits[idx],
        )


def predict_split(
    model: KeywordCNN,
    frontend: LogMelSpectrogram,
    audio: SplitAudio,
    rows: list[ManifestRow],
    labels: tuple[str, ...],
    *,
    batch_size: int,
) -> PredictionSet:
    """Eval-mode inference over ``audio`` in manifest order (no shuffling, no augmentation)."""
    if audio.relative_paths != tuple(r.relative_path for r in rows):
        raise PredictionError("audio and manifest rows are not in the same order")
    if any(r.split != audio.split for r in rows):
        raise PredictionError("rows from another split were passed")
    logits_all = infer_logits(
        model,
        frontend,
        audio.waveforms,
        batch_size=batch_size,
        num_classes=len(labels),
        to_float=pcm16_to_float,
    )
    return PredictionSet(
        split=audio.split,
        labels=labels,
        relative_paths=audio.relative_paths,
        audio_sha256=tuple(r.audio_sha256 for r in rows),
        speaker_ids=tuple(r.speaker_id for r in rows),
        y_true=audio.labels.numpy().astype(np.int64),
        logits=logits_all.astype(np.float32),
    )


def infer_logits(
    model: KeywordCNN,
    frontend: LogMelSpectrogram,
    windows: torch.Tensor,
    *,
    batch_size: int,
    num_classes: int,
    to_float: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> npt.NDArray[np.float32]:
    """Eval-mode logits for ``[N, num_samples]`` windows, in order, in fixed-size batches.

    ``to_float`` converts each batch to float32 (the clean path passes
    ``pcm16_to_float``); windows that are already float32 are used as they are.
    """
    model.eval()
    chunks: list[npt.NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start : start + batch_size]
            if to_float is not None:
                batch = to_float(batch)
            if batch.dtype != torch.float32:
                raise PredictionError(f"model input must be float32, got {batch.dtype}")
            logits = model(frontend(batch))
            chunks.append(logits.to(torch.float32).numpy().copy())
    if not chunks:
        return np.zeros((0, num_classes), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def prediction_columns(labels: tuple[str, ...]) -> list[str]:
    return [
        "relative_path",
        "audio_sha256",
        "speaker_id",
        "label",
        "label_index",
        "pred_index",
        "pred_label",
        "correct",
        *(f"logit_{label}" for label in labels),
        *(f"prob_{label}" for label in labels),
    ]


def _float(value: float) -> str:
    return repr(float(value))


def encode_predictions(predictions: PredictionSet) -> bytes:
    labels = predictions.labels
    probs = softmax(predictions.logits)
    y_pred = predictions.y_pred
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(prediction_columns(labels))
    for i, path in enumerate(predictions.relative_paths):
        truth = int(predictions.y_true[i])
        pred = int(y_pred[i])
        writer.writerow(
            [
                path,
                predictions.audio_sha256[i],
                predictions.speaker_ids[i],
                labels[truth],
                truth,
                pred,
                labels[pred],
                "true" if pred == truth else "false",
                *(_float(v) for v in predictions.logits[i]),
                *(_float(v) for v in probs[i]),
            ]
        )
    return buffer.getvalue().encode("utf-8")


def encode_calibrated(predictions: PredictionSet, temperature: float) -> bytes:
    labels = predictions.labels
    probs = softmax(predictions.logits, temperature)
    y_pred = predictions.y_pred
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["relative_path", "pred_index", *(f"cal_prob_{label}" for label in labels)])
    for i, path in enumerate(predictions.relative_paths):
        writer.writerow([path, int(y_pred[i]), *(_float(v) for v in probs[i])])
    return buffer.getvalue().encode("utf-8")


def decode_predictions(data: bytes, *, split: str, labels: tuple[str, ...]) -> PredictionSet:
    """Parse a prediction artifact and verify every derived column against the logits."""
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    header = next(reader, None)
    if header != prediction_columns(labels):
        raise PredictionError("prediction file header does not match the labels")
    records = list(reader)
    n_labels = len(labels)
    paths, hashes, speakers, truth = [], [], [], []
    logits = np.zeros((len(records), n_labels), dtype=np.float32)
    stored_pred, stored_correct = [], []
    stored_probs = np.zeros((len(records), n_labels), dtype=np.float64)
    for i, record in enumerate(records):
        if len(record) != len(header):
            raise PredictionError(f"row {i + 2}: wrong number of fields")
        path, digest, speaker, label, label_index, pred_index, pred_label, correct = record[:8]
        if labels[int(label_index)] != label or labels[int(pred_index)] != pred_label:
            raise PredictionError(f"row {i + 2}: label columns are inconsistent")
        paths.append(path)
        hashes.append(digest)
        speakers.append(speaker)
        truth.append(int(label_index))
        stored_pred.append(int(pred_index))
        stored_correct.append(correct)
        values = [float(v) for v in record[8 : 8 + n_labels]]
        logits[i] = np.asarray(values, dtype=np.float32)
        if [float(np.float32(v)) for v in values] != values:
            raise PredictionError(f"row {i + 2}: logits are not float32 values")
        stored_probs[i] = [float(v) for v in record[8 + n_labels :]]
    result = PredictionSet(
        split=split,
        labels=labels,
        relative_paths=tuple(paths),
        audio_sha256=tuple(hashes),
        speaker_ids=tuple(speakers),
        y_true=np.asarray(truth, dtype=np.int64),
        logits=logits,
    )
    if result.y_pred.tolist() != stored_pred:
        raise PredictionError("stored pred_index disagrees with the argmax of the stored logits")
    recomputed_correct = [
        "true" if p == t else "false" for p, t in zip(stored_pred, truth, strict=True)
    ]
    if recomputed_correct != stored_correct:
        raise PredictionError("stored correct column disagrees with the labels")
    if not np.array_equal(softmax(logits), stored_probs):
        raise PredictionError("stored probabilities disagree with softmax of the stored logits")
    return result
