"""Local streaming demo (docs/demo.md).

:func:`run_demo` takes one local 16 kHz mono 16-bit PCM WAV file and:

1. loads the pinned Milestone 6 ONNX export through :func:`load_verified_export`
   (report SHA-256 pinned, passed parity, checkpoint, labels, features, and
   manifest checked) and opens an ONNX Runtime CPU session;
2. pushes the audio through the ``kw10-streaming-v1`` streaming classifier in
   100 ms chunks, like a live source, recording each window's prediction,
   confidence, causal aggregate, and ONNX Runtime inference time;
3. optionally applies one ``kw10-robustness-v1`` corruption and severity to the
   whole file (seeded from the audio content) and streams the result the same
   way, so the clean and corrupted outputs can be compared;
4. writes a JSON record and a self-contained HTML page (no scripts, no external
   resources) into a git-ignored folder.

Nothing is uploaded and no network service is used. Demo outputs are not
measured results: they describe one file on one machine. The model is a
closed-set 10-keyword classifier, so every window gets one of the 10 labels,
including silence, noise, and other speech.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import DatasetSpec
from robust_streaming_ml.data.wav import WavError, parse_wav
from robust_streaming_ml.demo.config import DemoConfig
from robust_streaming_ml.export.onnx_model import OnnxWindowBackend, create_session
from robust_streaming_ml.export.run import load_verified_export
from robust_streaming_ml.robustness.corruptions import (
    PROTOCOL_ID as CORRUPTION_PROTOCOL_ID,
)
from robust_streaming_ml.robustness.corruptions import (
    CorruptionError,
    CorruptionSpec,
    Severity,
    corrupt,
    get_corruption,
)
from robust_streaming_ml.run_dir import ProvenanceError, current_source_revision
from robust_streaming_ml.streaming.classifier import (
    StreamDecision,
    StreamingClassifier,
    WindowBackend,
    WindowPrediction,
)
from robust_streaming_ml.streaming.windowing import StreamingSpec

__all__ = [
    "LIMITATIONS",
    "DemoAudio",
    "DemoError",
    "DemoResult",
    "read_demo_wav",
    "run_demo",
    "stream_file",
]

LIMITATIONS = (
    "Demo output for one local file on one machine: not a measured or reported result.",
    "Not production-ready, and no reliability or real-time claim.",
    "Closed set: every window is assigned one of the 10 keywords, including silence, noise, "
    "and other speech. There is no silence or unknown-word rejection.",
    "Trained and evaluated on isolated 1 s Speech Commands clips (one model, one seed); "
    "behaviour on continuous or real-world audio was not evaluated.",
    "Corruptions are the synthetic kw10-robustness-v1 transforms, applied here to the whole "
    "file as one clip; they are not real-world recordings.",
    "Timings were measured during this demo run; the protocol benchmark is kw10-latency-v1 "
    "in RESULTS.md.",
    "The input must be 16 kHz mono 16-bit PCM WAV; nothing is resampled or converted.",
)

Int16 = npt.NDArray[np.int16]
Float32 = npt.NDArray[np.float32]


class DemoError(RuntimeError):
    """The input or the pinned model is not usable by the demo."""


@dataclass(frozen=True)
class DemoAudio:
    path: Path
    data: bytes
    file_sha256: str
    samples: Int16
    sample_rate_hz: int

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.sample_rate_hz

    @property
    def audio_sha256(self) -> str:
        return hashlib.sha256(self.samples.astype("<i2").tobytes()).hexdigest()


@dataclass(frozen=True)
class StreamOutcome:
    predictions: list[WindowPrediction]
    decision: StreamDecision | None
    wall_ms: float


@dataclass(frozen=True)
class DemoResult:
    record: dict[str, Any]
    json_path: Path
    html_path: Path


def read_demo_wav(path: Path, *, sample_rate_hz: int, max_seconds: float) -> DemoAudio:
    """Read a local WAV file that matches the model's input contract exactly."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DemoError(f"cannot read {path}: {exc}") from exc
    try:
        wav = parse_wav(data)
    except WavError as exc:
        raise DemoError(f"{path} is not a valid WAV file: {exc}") from exc
    found = (wav.num_channels, wav.sample_rate_hz, wav.sample_format)
    if found != (1, sample_rate_hz, "PCM_16"):
        raise DemoError(
            f"{path}: the model needs mono {sample_rate_hz} Hz 16-bit PCM WAV, got "
            f"{wav.num_channels} channel(s), {wav.sample_rate_hz} Hz, {wav.sample_format}. "
            "Convert it first; the demo does not resample or mix channels."
        )
    if wav.num_frames == 0:
        raise DemoError(f"{path} contains no audio")
    if wav.num_frames > max_seconds * sample_rate_hz:
        raise DemoError(
            f"{path} is {wav.num_frames / sample_rate_hz:.1f} s long; the limit is "
            f"{max_seconds:g} s (demo.max_seconds)"
        )
    return DemoAudio(
        path=path,
        data=data,
        file_sha256=hashlib.sha256(data).hexdigest(),
        samples=wav.samples_int16().copy(),
        sample_rate_hz=wav.sample_rate_hz,
    )


def stream_file(
    backend: WindowBackend,
    spec: StreamingSpec,
    labels: Sequence[str],
    samples: npt.NDArray[Any],
) -> StreamOutcome:
    """Push the audio in hop-sized chunks (like a live source), then flush."""
    classifier = StreamingClassifier(
        backend, spec, labels, sample_rate_hz=spec.sample_rate_hz, measure_time=True
    )
    started = time.perf_counter()
    predictions: list[WindowPrediction] = []
    for start in range(0, samples.shape[0], spec.hop_samples):
        predictions.extend(classifier.push(samples[start : start + spec.hop_samples]))
    predictions.extend(classifier.flush())
    wall_ms = (time.perf_counter() - started) * 1e3
    return StreamOutcome(predictions=predictions, decision=classifier.decision(), wall_ms=wall_ms)


def _decision(decision: StreamDecision | None, labels: Sequence[str]) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "label": decision.label,
        "confidence": decision.confidence,
        "windows": decision.windows,
        "rule": "aggregate of the last emitted window (mean of up to 5 window probabilities)",
        "probabilities": {
            label: float(p) for label, p in zip(labels, decision.probabilities, strict=True)
        },
    }


def _outcome(outcome: StreamOutcome, labels: Sequence[str], rate: int) -> dict[str, Any]:
    inference = [p.inference_ns / 1e6 for p in outcome.predictions if p.inference_ns is not None]
    return {
        "windows": [
            {
                **p.describe(labels, rate),
                "probabilities": [float(v) for v in p.probabilities],
                "aggregate_probabilities": [float(v) for v in p.aggregate_probabilities],
            }
            for p in outcome.predictions
        ],
        "decision": _decision(outcome.decision, labels),
        "timing": {
            "note": "measured during this demo run on this machine; not the protocol benchmark",
            "inference_ms_median": float(np.median(inference)) if inference else None,
            "inference_ms_max": float(np.max(inference)) if inference else None,
            "windows_timed": len(inference),
            "stream_wall_ms": outcome.wall_ms,
        },
    }


def _source() -> dict[str, Any]:
    try:
        source = current_source_revision()
    except ProvenanceError as exc:
        return {"unavailable": str(exc).splitlines()[0]}
    return {"commit": source.commit, "dirty": source.dirty}


def run_demo(
    config: DemoConfig,
    audio_path: Path,
    *,
    corruption: str | None = None,
    severity: str | None = None,
    output: Path | None = None,
    dataset_spec: DatasetSpec = SPEECH_COMMANDS_KW10,
) -> DemoResult:
    """Run the demo on one file; see the module docstring."""
    from robust_streaming_ml.demo.report import render_html

    if (corruption is None) != (severity is None):
        raise DemoError("give both --corruption and --severity, or neither")
    chosen: tuple[CorruptionSpec, Severity] | None = None
    if corruption is not None and severity is not None:
        try:
            corruption_spec = get_corruption(corruption)
            chosen = (corruption_spec, corruption_spec.severity(severity))
        except CorruptionError as exc:
            raise DemoError(str(exc)) from exc
    model, export_report = load_verified_export(
        config.export.run_dir,
        report_sha256=config.export.report_sha256,
        expect_labels=dataset_spec.labels,
    )
    expected_manifest = {
        "dataset": dataset_spec.manifest_name,
        "sha256": dataset_spec.expected_manifest_sha256,
    }
    if model.metadata["manifest"] != expected_manifest:
        raise DemoError(
            f"the model records manifest {model.metadata['manifest']}, not {expected_manifest}"
        )
    import onnxruntime

    session = create_session(model, intra_op_threads=config.demo.intra_op_threads)
    backend = OnnxWindowBackend(session)
    spec = StreamingSpec.from_window(model.window)
    labels = model.labels
    rate = spec.sample_rate_hz
    audio = read_demo_wav(audio_path, sample_rate_hz=rate, max_seconds=config.demo.max_seconds)

    clean = stream_file(backend, spec, labels, audio.samples)
    corrupted_record: dict[str, Any] | None = None
    corrupted_outcome: StreamOutcome | None = None
    corrupted_samples: Float32 | None = None
    if chosen is not None:
        corruption_spec, corruption_severity = chosen
        identity = f"demo:{audio.audio_sha256}"
        try:
            clip = corrupt(corruption_spec, corruption_severity, audio.samples, identity)
        except CorruptionError as exc:
            raise DemoError(f"corruption failed: {exc}") from exc
        corrupted_samples = clip.samples
        corrupted_outcome = stream_file(backend, spec, labels, clip.samples)
        corrupted_record = {
            "protocol": CORRUPTION_PROTOCOL_ID,
            "name": corruption_spec.name,
            "version": corruption_spec.version,
            "severity": corruption_severity.name,
            "params": dict(corruption_severity.params),
            "description": corruption_spec.description,
            "identity": identity,
            "seed": clip.seed,
            "measures": clip.measures,
            "applied_to": "the whole file as one clip",
        }

    archive = dataset_spec.archive
    record: dict[str, Any] = {
        "demo": "kw10 streaming keyword-spotting demo (local, offline)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "not_a_result": LIMITATIONS[0],
        "source": _source(),
        "model": {
            "onnx_sha256": model.sha256,
            "export_run_id": export_report["export_run_id"],
            "export_report_sha256": config.export.report_sha256,
            "export_parity_passed": export_report["passed"],
            "checkpoint": model.metadata["checkpoint"],
            "labels": list(labels),
            "opset": model.metadata["contract"]["opset"],
            "feature_fingerprint": model.metadata["preprocessing"]["feature_fingerprint"],
            "runtime": {
                "onnxruntime": str(onnxruntime.__version__),
                "providers": list(session.get_providers()),
                "intra_op_threads": config.demo.intra_op_threads,
            },
        },
        "dataset": {
            "name": archive.name,
            "version": archive.version,
            "subset": dataset_spec.manifest_name,
            "manifest_sha256": model.metadata["manifest"]["sha256"],
            "license": dataset_spec.metadata.get("license_name"),
            "attribution": dataset_spec.metadata.get("attribution_text"),
            "citation": dataset_spec.metadata.get("citation"),
        },
        "streaming": spec.describe(),
        "config": config.to_dict(),
        "input": {
            "file": audio_path.name,
            "file_sha256": audio.file_sha256,
            "audio_sha256": audio.audio_sha256,
            "sample_rate_hz": rate,
            "samples": int(audio.samples.size),
            "duration_s": audio.duration_s,
        },
        "corruption": corrupted_record,
        "runs": {
            "clean": _outcome(clean, labels, rate),
            "corrupted": (_outcome(corrupted_outcome, labels, rate) if corrupted_outcome else None),
        },
        "limitations": list(LIMITATIONS),
    }
    if corrupted_outcome is not None:
        same_windows = sum(
            a.label_index == b.label_index
            for a, b in zip(clean.predictions, corrupted_outcome.predictions, strict=True)
        )
        record["comparison"] = {
            "windows": len(clean.predictions),
            "same_window_label": same_windows,
            "same_decision": (
                clean.decision is not None
                and corrupted_outcome.decision is not None
                and clean.decision.label_index == corrupted_outcome.decision.label_index
            ),
        }

    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{corruption}_{severity}" if corruption else ""
        output = config.demo.output_dir / f"{audio_path.stem}{suffix}_{stamp}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    page = render_html(
        record,
        clean_wav=audio.data if config.demo.embed_audio else None,
        corrupted_samples=corrupted_samples if config.demo.embed_audio else None,
        sample_rate_hz=rate,
    )
    output.write_text(page, encoding="utf-8")
    return DemoResult(record=record, json_path=json_path, html_path=output)
