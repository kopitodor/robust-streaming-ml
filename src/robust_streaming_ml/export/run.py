"""Export and parity entry point (protocols ``kw10-onnx-v1`` and ``kw10-streaming-v1``).

Before anything is written, :func:`export_and_verify` checks:

1. the pinned checkpoint and its provenance (the Milestone 4 gate);
2. the validated manifest;
3. the pinned Milestone 4 baseline: report SHA-256, protocol, checkpoint,
   manifest, labels, status, batch size, threads, and test predictions;
4. the exact export toolchain, and ONNX Runtime at the pinned version with
   the CPU provider;
5. a clean source tree.

It then calls ``start_run``. Inside the run it:

1. exports the model twice (the bytes must be identical), writes
   ``onnx/model.onnx`` and its sidecar, reloads the file with every check,
   and opens a CPU session with one intra-op thread;
2. loads the official test split and requires the PyTorch path to reproduce
   the Milestone 4 raw test logits bit-for-bit (batch size 256);
3. parity set A: every test clip is its own stream, pushed in seeded chunks
   through the PyTorch and ONNX streaming classifiers. Each yields exactly
   one window, which must equal the Milestone 4 clip window bit-for-bit;
4. parity set B: 510 streams, each concatenating 8 consecutive test clips
   (the last has 2). Every streamed window must equal the offline reference
   window bit-for-bit;
5. compares logits, probabilities, and aggregates against the predeclared
   tolerances, measures streaming consistency against the Milestone 4
   predictions, and writes ``parity/*.csv``, ``report.json``, ``report.md``,
   and ``summary.json``.

A tolerance violation does not stop the run: every artifact and the report,
listing the violations, are still written, ``summary.json`` has status
``failed``, and :attr:`ExportResult.passed` is false. Any other failure
raises, and ``summary.json`` records it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from robust_streaming_ml.checkpoint import LoadedCheckpoint
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import DatasetSpec
from robust_streaming_ml.evaluation.predictions import (
    PredictionSet,
    decode_predictions,
    infer_logits,
)
from robust_streaming_ml.evaluation.run import verify_checkpoint
from robust_streaming_ml.export.config import ExportConfig
from robust_streaming_ml.export.onnx_model import (
    ONNX_PROTOCOL_ID,
    LoadedOnnxModel,
    OnnxModelError,
    OnnxWindowBackend,
    create_session,
    export_onnx_bytes,
    graph_summary,
    installed_versions,
    load_onnx_model,
    model_metadata,
    require_runtime,
    require_toolchain,
    session_options_record,
    write_onnx_model,
)
from robust_streaming_ml.export.parity import (
    compare_aggregates,
    compare_logits,
    tolerances,
)
from robust_streaming_ml.run_dir import (
    ProvenanceError,
    RunDirectory,
    current_source_revision,
    start_run,
)
from robust_streaming_ml.seed import derive_seed
from robust_streaming_ml.streaming.classifier import (
    StreamingClassifier,
    TorchWindowBackend,
    WindowPrediction,
)
from robust_streaming_ml.streaming.windowing import (
    StreamingSpec,
    offline_windows,
    stream_samples,
)
from robust_streaming_ml.training.config import FeatureSection, WindowSection
from robust_streaming_ml.training.data import (
    ValidatedManifest,
    check_manifest_inputs,
    read_split_audio,
)

__all__ = [
    "BASELINE_PROTOCOL",
    "MAX_CHUNK",
    "ORT_PARITY_THREADS",
    "PARITY_SEED",
    "STREAM_GROUP_SIZE",
    "ExportError",
    "ExportResult",
    "chunk_sizes",
    "export_and_verify",
    "load_verified_export",
    "split_chunks",
]

BASELINE_PROTOCOL = "kw10-clean-eval-v1"
PARITY_SEED = 2718
STREAM_GROUP_SIZE = 8
MAX_CHUNK = 32000
ORT_PARITY_THREADS = 1
SPLIT = "test"
REPORT_SCHEMA_VERSION = 1
MODEL_FILE = "onnx/model.onnx"

Int16 = npt.NDArray[np.int16]
Float32 = npt.NDArray[np.float32]


class ExportError(RuntimeError):
    """A gate failed or the export run cannot be completed as specified."""


@dataclass(frozen=True)
class Baseline:
    run_id: str
    report_sha256: str
    test_accuracy: float
    test_predictions: PredictionSet
    test_predictions_sha256: str


@dataclass(frozen=True)
class ExportResult:
    run: RunDirectory
    report: dict[str, Any]
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.report["passed"])


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


def chunk_sizes(stream_id: str, total: int) -> list[int]:
    """Seeded log-uniform chunk sizes in ``[1, MAX_CHUNK]`` that sum to ``total``."""
    rng = np.random.default_rng(derive_seed(PARITY_SEED, ONNX_PROTOCOL_ID, "chunks", stream_id))
    sizes: list[int] = []
    remaining = total
    upper = float(np.log(MAX_CHUNK + 1))
    while remaining > 0:
        size = min(int(np.floor(np.exp(rng.uniform(0.0, upper)))), MAX_CHUNK, remaining)
        size = max(size, 1)
        sizes.append(size)
        remaining -= size
    return sizes


def split_chunks(samples: npt.NDArray[Any], sizes: Sequence[int]) -> list[npt.NDArray[Any]]:
    if sum(sizes) != samples.shape[0]:
        raise ExportError("chunk sizes do not cover the stream")
    return list(np.split(samples, np.cumsum(sizes)[:-1])) if sizes else []


def _check_baseline(
    config: ExportConfig, loaded: LoadedCheckpoint, manifest: ValidatedManifest
) -> Baseline:
    run_dir = config.baseline.run_dir
    try:
        report_bytes = (run_dir / "report.json").read_bytes()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExportError(f"baseline evaluation run unreadable: {exc}") from exc
    digest = _sha256(report_bytes)
    if digest != config.baseline.report_sha256:
        raise ExportError(
            f"baseline report SHA-256 {digest} != pinned {config.baseline.report_sha256}"
        )
    report = json.loads(report_bytes)
    checks = {
        "protocol": (report["protocol"]["id"], BASELINE_PROTOCOL),
        "checkpoint": (report["checkpoint"]["sha256"], loaded.sha256),
        "manifest": (report["manifest"]["sha256"], manifest.sha256),
        "labels": (report["labels"], list(manifest.labels)),
        "status": (summary.get("status"), "completed"),
        "batch_size": (report["batch_size"], config.export.reference_batch_size),
    }
    if config.export.num_threads is not None:
        checks["torch_num_threads"] = (report["torch_num_threads"], config.export.num_threads)
    for name, (found, expected) in checks.items():
        if found != expected:
            raise ExportError(f"baseline {name} {found!r} != expected {expected!r}")
    test_bytes = (run_dir / "predictions" / "test.csv").read_bytes()
    test_sha = _sha256(test_bytes)
    if test_sha != report["artifacts_sha256"]["predictions/test.csv"]:
        raise ExportError("baseline test predictions do not match the baseline report")
    return Baseline(
        run_id=report["evaluation_run_id"],
        report_sha256=digest,
        test_accuracy=float(report["metrics"][SPLIT]["raw"]["accuracy"]),
        test_predictions=decode_predictions(test_bytes, split=SPLIT, labels=manifest.labels),
        test_predictions_sha256=test_sha,
    )


def export_and_verify(
    config: ExportConfig,
    *,
    dataset_spec: DatasetSpec = SPEECH_COMMANDS_KW10,
    root: Path | None = None,
    run_id: str | None = None,
    allow_dirty_source: bool = False,
) -> ExportResult:
    """Run the protocol; see the module docstring."""
    loaded = verify_checkpoint(config.checkpoint, config.data)
    window = WindowSection.model_validate(loaded.metadata["preprocessing"]["window"])
    manifest = check_manifest_inputs(config.data, window, dataset_spec)
    baseline = _check_baseline(config, loaded, manifest)
    toolchain = require_toolchain()
    runtime = require_runtime()
    source = current_source_revision()
    if source.dirty and not allow_dirty_source:
        raise ProvenanceError(
            f"working tree has uncommitted changes at {source.commit}; commit them "
            "(or pass allow_dirty_source for a non-result development run)"
        )
    run = start_run(config, root=root, run_id=run_id)
    try:
        report, summary = _run(config, run, loaded, manifest, window, baseline, toolchain, runtime)
    except BaseException as exc:
        _write_atomic(
            run.path / "summary.json",
            _json_bytes(
                {"status": "failed", "run_id": run.run_id, "error": f"{type(exc).__name__}: {exc}"}
            ),
        )
        raise
    return ExportResult(run=run, report=report, summary=summary)


def _stream(
    classifier: StreamingClassifier, chunks: list[npt.NDArray[Any]]
) -> list[WindowPrediction]:
    classifier.reset()
    predictions: list[WindowPrediction] = []
    for chunk in chunks:
        predictions.extend(classifier.push(chunk))
    predictions.extend(classifier.flush())
    return predictions


def _float(value: float) -> str:
    return repr(float(value))


def _csv(header: list[str], rows: list[list[Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _same_window(a: WindowPrediction, b: Any) -> bool:
    return bool(
        a.window.index == b.index
        and a.window.start_sample == b.start_sample
        and a.window.valid_samples == b.valid_samples
        and a.window.final_partial == b.final_partial
        and a.window.emitted_at_sample == b.emitted_at_sample
        and np.array_equal(a.window.samples, b.samples)
    )


def _run(
    config: ExportConfig,
    run: RunDirectory,
    loaded: LoadedCheckpoint,
    manifest: ValidatedManifest,
    window: WindowSection,
    baseline: Baseline,
    toolchain: dict[str, str],
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    if config.export.num_threads is not None:
        torch.set_num_threads(config.export.num_threads)
    labels = manifest.labels
    num_classes = len(labels)
    spec = StreamingSpec.from_window(window)
    prep = loaded.metadata["preprocessing"]
    artifacts: dict[str, str] = {}

    # 1. export twice, write, reload with every check, open the CPU session
    metadata = model_metadata(loaded, source=run.metadata["source"], toolchain=toolchain)
    exported = export_onnx_bytes(loaded, metadata)
    if export_onnx_bytes(loaded, metadata) != exported:
        raise OnnxModelError("exporting the same checkpoint twice gave different bytes")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": installed_versions(),
    }
    model_sha = write_onnx_model(
        run.path / MODEL_FILE,
        exported,
        run_id=run.run_id,
        created_at_utc=run.metadata["created_at_utc"],
        environment=environment,
    )
    artifacts[MODEL_FILE] = model_sha
    artifacts[MODEL_FILE + ".json"] = _sha256((run.path / (MODEL_FILE + ".json")).read_bytes())
    onnx_model = load_onnx_model(
        run.path / MODEL_FILE,
        expect_checkpoint_sha256=loaded.sha256,
        expect_labels=labels,
        expect_feature_fingerprint=prep["feature_fingerprint"],
        expect_window=window,
        expect_features=FeatureSection.model_validate(prep["features"]),
        expect_manifest_sha256=manifest.sha256,
    )
    session = create_session(onnx_model, intra_op_threads=ORT_PARITY_THREADS)
    torch_cls = StreamingClassifier(
        TorchWindowBackend(loaded.model, loaded.frontend),
        spec,
        labels,
        sample_rate_hz=window.sample_rate_hz,
    )
    onnx_cls = StreamingClassifier(
        OnnxWindowBackend(session), spec, labels, sample_rate_hz=window.sample_rate_hz
    )

    # 2. test audio and the clean-path gate
    rows = manifest.rows_for(SPLIT)
    audio = read_split_audio(manifest, SPLIT, config.data, window)
    if baseline.test_predictions.relative_paths != audio.relative_paths:
        raise ExportError("baseline test predictions are not in manifest order")
    clips: list[Int16] = [audio.waveforms[i, : r.num_frames].numpy() for i, r in enumerate(rows)]
    clip_windows = np.zeros((len(clips), spec.window_samples), dtype=np.float32)
    for i, clip in enumerate(clips):
        clip_windows[i, : clip.size] = stream_samples(clip)
    gate_logits = infer_logits(
        loaded.model,
        loaded.frontend,
        torch.from_numpy(clip_windows),
        batch_size=config.export.reference_batch_size,
        num_classes=num_classes,
    )
    if not np.array_equal(gate_logits, baseline.test_predictions.logits):
        worst = float(np.max(np.abs(gate_logits - baseline.test_predictions.logits)))
        raise ExportError(
            f"PyTorch path does not reproduce the Milestone 4 test logits (max |diff| {worst})"
        )

    # 3. set A: every clip is its own stream
    n_clips = len(clips)
    a_torch = np.zeros((n_clips, num_classes), dtype=np.float32)
    a_onnx = np.zeros_like(a_torch)
    a_torch_agg = np.zeros((n_clips, num_classes), dtype=np.float64)
    a_onnx_agg = np.zeros_like(a_torch_agg)
    torch_decisions = np.zeros(n_clips, dtype=np.int64)
    onnx_decisions = np.zeros(n_clips, dtype=np.int64)
    valid = np.zeros(n_clips, dtype=np.int64)
    for i, (row, clip) in enumerate(zip(rows, clips, strict=True)):
        chunks = split_chunks(clip, chunk_sizes(row.relative_path, clip.size))
        by_backend = {"torch": _stream(torch_cls, chunks), "onnx": _stream(onnx_cls, chunks)}
        for name, predictions in by_backend.items():
            if len(predictions) != 1 or not np.array_equal(
                predictions[0].window.samples, clip_windows[i]
            ):
                raise ExportError(
                    f"{row.relative_path}: {name} streaming did not yield exactly the clip-level "
                    "window"
                )
        torch_pred, onnx_pred = by_backend["torch"][0], by_backend["onnx"][0]
        torch_decision, onnx_decision = torch_cls.decision(), onnx_cls.decision()
        if torch_decision is None or onnx_decision is None:
            raise ExportError(f"{row.relative_path}: no stream decision")
        a_torch[i], a_onnx[i] = torch_pred.logits, onnx_pred.logits
        a_torch_agg[i] = torch_pred.aggregate_probabilities
        a_onnx_agg[i] = onnx_pred.aggregate_probabilities
        torch_decisions[i] = torch_decision.label_index
        onnx_decisions[i] = onnx_decision.label_index
        valid[i] = torch_pred.window.valid_samples
    clip_ids = [r.relative_path for r in rows]
    parity_a: dict[str, Any] = {
        "streams": n_clips,
        "windows_identical_to_clip_level": True,
        "logits": compare_logits(a_torch, a_onnx, row_ids=clip_ids),
        "aggregates": compare_aggregates(a_torch_agg, a_onnx_agg, row_ids=clip_ids),
    }
    m4_pred = baseline.test_predictions.y_pred
    y_true = baseline.test_predictions.y_true
    consistency: dict[str, Any] = {
        "clips": n_clips,
        "definition": "stream decision (kw10-streaming-v1) == Milestone 4 clip-level prediction",
        "milestone4_test_accuracy": baseline.test_accuracy,
        **{
            name: {
                "agree_with_milestone4": int(np.sum(decisions == m4_pred)),
                "agreement_rate": float(np.mean(decisions == m4_pred)),
                "accuracy": float(np.mean(decisions == y_true)),
                "correct": int(np.sum(decisions == y_true)),
            }
            for name, decisions in (("onnxruntime", onnx_decisions), ("pytorch", torch_decisions))
        },
    }
    artifacts["parity/clips.csv"] = _write_atomic(
        run.path / "parity" / "clips.csv",
        _csv(
            [
                "relative_path",
                "label_index",
                "milestone4_pred_index",
                "pytorch_decision_index",
                "onnx_decision_index",
                "valid_samples",
                *(f"torch_logit_{label}" for label in labels),
                *(f"onnx_logit_{label}" for label in labels),
            ],
            [
                [
                    clip_ids[i],
                    int(y_true[i]),
                    int(m4_pred[i]),
                    int(torch_decisions[i]),
                    int(onnx_decisions[i]),
                    int(valid[i]),
                    *(_float(v) for v in a_torch[i]),
                    *(_float(v) for v in a_onnx[i]),
                ]
                for i in range(n_clips)
            ],
        ),
    )

    # 4. set B: concatenation streams, compared with the offline reference windows
    b_torch: list[Float32] = []
    b_onnx: list[Float32] = []
    b_torch_agg: list[npt.NDArray[np.float64]] = []
    b_onnx_agg: list[npt.NDArray[np.float64]] = []
    window_rows: list[list[Any]] = []
    stream_rows: list[list[Any]] = []
    window_ids: list[str] = []
    total_chunks = 0
    for index, start in enumerate(range(0, n_clips, STREAM_GROUP_SIZE)):
        members = list(range(start, min(start + STREAM_GROUP_SIZE, n_clips)))
        stream_id = f"stream-{index}"
        signal = np.concatenate([clips[j] for j in members])
        sizes = chunk_sizes(stream_id, int(signal.size))
        chunks = split_chunks(signal, sizes)
        total_chunks += len(sizes)
        streamed_torch = _stream(torch_cls, chunks)
        streamed_onnx = _stream(onnx_cls, chunks)
        reference = offline_windows(signal, spec)
        if not (len(streamed_torch) == len(streamed_onnx) == len(reference)) or not all(
            _same_window(t, r) and _same_window(o, r)
            for t, o, r in zip(streamed_torch, streamed_onnx, reference, strict=True)
        ):
            raise ExportError(f"{stream_id}: streamed windows differ from the offline reference")
        for t, o in zip(streamed_torch, streamed_onnx, strict=True):
            b_torch.append(t.logits)
            b_onnx.append(o.logits)
            b_torch_agg.append(t.aggregate_probabilities)
            b_onnx_agg.append(o.aggregate_probabilities)
            window_ids.append(f"{stream_id}/{t.window.index}")
            w = t.window
            window_rows.append(
                [
                    index,
                    w.index,
                    w.start_sample,
                    w.valid_samples,
                    "true" if w.final_partial else "false",
                    w.emitted_at_sample,
                    *(_float(v) for v in t.logits),
                    *(_float(v) for v in o.logits),
                ]
            )
        stream_rows.append(
            [
                index,
                stream_id,
                len(members),
                "|".join(clip_ids[j] for j in members),
                int(signal.size),
                len(reference),
                len(sizes),
            ]
        )
    b_torch_arr, b_onnx_arr = np.stack(b_torch), np.stack(b_onnx)
    parity_b: dict[str, Any] = {
        "streams": len(stream_rows),
        "clips_per_stream": STREAM_GROUP_SIZE,
        "windows_identical_to_offline": True,
        "chunks": total_chunks,
        "logits": compare_logits(b_torch_arr, b_onnx_arr, row_ids=window_ids),
        "aggregates": compare_aggregates(
            np.stack(b_torch_agg), np.stack(b_onnx_agg), row_ids=window_ids
        ),
        "final_partial_windows": sum(1 for r in window_rows if r[4] == "true"),
    }
    artifacts["parity/stream_windows.csv"] = _write_atomic(
        run.path / "parity" / "stream_windows.csv",
        _csv(
            [
                "stream_index",
                "window_index",
                "start_sample",
                "valid_samples",
                "final_partial",
                "emitted_at_sample",
                *(f"torch_logit_{label}" for label in labels),
                *(f"onnx_logit_{label}" for label in labels),
            ],
            window_rows,
        ),
    )
    artifacts["parity/streams.csv"] = _write_atomic(
        run.path / "parity" / "streams.csv",
        _csv(
            [
                "stream_index",
                "stream_id",
                "clips",
                "relative_paths",
                "num_samples",
                "num_windows",
                "num_chunks",
            ],
            stream_rows,
        ),
    )

    # 5. report
    passed = all(
        part[key]["passed"] for part in (parity_a, parity_b) for key in ("logits", "aggregates")
    )
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "protocols": {"export": ONNX_PROTOCOL_ID, "streaming": spec.describe()},
        "export_run_id": run.run_id,
        "source": run.metadata["source"],
        "checkpoint": {
            "path": config.checkpoint.path.as_posix(),
            "sha256": loaded.sha256,
            "run_id": loaded.metadata["run_id"],
            "epoch": loaded.metadata["epoch"],
            "feature_fingerprint": prep["feature_fingerprint"],
        },
        "manifest": {"sha256": manifest.sha256, "test_rows": n_clips},
        "labels": list(labels),
        "baseline": {
            "evaluation_run_id": baseline.run_id,
            "report_sha256": baseline.report_sha256,
            "test_predictions_sha256": baseline.test_predictions_sha256,
        },
        "toolchain": toolchain,
        "runtime": runtime,
        "session_options": session_options_record(intra_op_threads=ORT_PARITY_THREADS),
        "torch_num_threads": torch.get_num_threads(),
        "onnx_model": {
            "path": MODEL_FILE,
            "sha256": model_sha,
            "bytes": len(exported),
            "deterministic_export": True,
            "graph": graph_summary(_proto(onnx_model)),
        },
        "clean_path_check": {
            "reproduced_baseline_test_logits": True,
            "batch_size": config.export.reference_batch_size,
            "examples": n_clips,
        },
        "tolerances": tolerances(),
        "parity": {"clip_streams": parity_a, "concatenation_streams": parity_b},
        "streaming_consistency": consistency,
        "passed": passed,
        "seeds": {
            "chunk_seed": f"derive_seed({PARITY_SEED}, {ONNX_PROTOCOL_ID!r}, 'chunks', stream_id)",
            "max_chunk": MAX_CHUNK,
        },
        "artifacts_sha256": artifacts,
    }
    all_artifacts = dict(artifacts)
    all_artifacts["report.json"] = _write_atomic(run.path / "report.json", _json_bytes(report))
    all_artifacts["report.md"] = _write_atomic(run.path / "report.md", _markdown(report))
    summary = {
        "status": "completed" if passed else "failed",
        "run_id": run.run_id,
        "protocol": ONNX_PROTOCOL_ID,
        "parity_passed": passed,
        "onnx_sha256": model_sha,
        "headline": {
            "clip_streams_max_abs_logit_diff": parity_a["logits"]["abs_logit_diff"]["max"],
            "concatenation_windows": parity_b["logits"]["windows"],
            "concatenation_max_abs_logit_diff": parity_b["logits"]["abs_logit_diff"]["max"],
            "onnx_stream_decision_agreement": consistency["onnxruntime"]["agreement_rate"],
        },
        "artifacts_sha256": all_artifacts,
        "seconds": round(time.perf_counter() - started, 3),
    }
    if not passed:
        summary["error"] = "parity tolerances violated; see report.json"
    _write_atomic(run.path / "summary.json", _json_bytes(summary))
    return report, summary


def _proto(model: LoadedOnnxModel) -> Any:
    import onnx

    return onnx.load_model_from_string(model.data)


def load_verified_export(
    run_dir: Path,
    *,
    report_sha256: str,
    expect_checkpoint_sha256: str | None = None,
    expect_labels: Sequence[str] | None = None,
) -> tuple[LoadedOnnxModel, dict[str, Any]]:
    """An exported model whose pinned export report records passed parity."""
    try:
        report_bytes = (run_dir / "report.json").read_bytes()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExportError(f"export run unreadable: {exc}") from exc
    digest = _sha256(report_bytes)
    if digest != report_sha256:
        raise ExportError(f"export report SHA-256 {digest} != pinned {report_sha256}")
    report = json.loads(report_bytes)
    if report.get("protocols", {}).get("export") != ONNX_PROTOCOL_ID:
        raise ExportError("export report is not protocol kw10-onnx-v1")
    if report.get("passed") is not True or summary.get("status") != "completed":
        raise ExportError("the pinned export did not pass parity; it cannot be used")
    model = load_onnx_model(
        run_dir / MODEL_FILE,
        expect_checkpoint_sha256=expect_checkpoint_sha256 or report["checkpoint"]["sha256"],
        expect_labels=list(expect_labels) if expect_labels is not None else report["labels"],
        expect_feature_fingerprint=report["checkpoint"]["feature_fingerprint"],
        expect_manifest_sha256=report["manifest"]["sha256"],
    )
    if model.sha256 != report["onnx_model"]["sha256"]:
        raise ExportError("ONNX model SHA-256 differs from the export report")
    return model, report


def _markdown(report: dict[str, Any]) -> bytes:
    a = report["parity"]["clip_streams"]
    b = report["parity"]["concatenation_streams"]
    tol = report["tolerances"]
    lines = [
        f"# Export and parity report: {report['protocols']['export']}",
        "",
        f"- Run `{report['export_run_id']}` at `{report['source']['commit']}` "
        f"(dirty {report['source']['dirty']})",
        f"- Checkpoint `{report['checkpoint']['sha256']}`; manifest "
        f"`{report['manifest']['sha256']}`",
        f"- ONNX model `{report['onnx_model']['path']}` SHA-256 `{report['onnx_model']['sha256']}` "
        f"({report['onnx_model']['bytes']} bytes, opset "
        f"{report['onnx_model']['graph']['opset_imports']}); exported twice, identical bytes",
        f"- Toolchain {report['toolchain']}; runtime {report['runtime']['onnxruntime']} "
        f"{report['runtime']['provider']}; session {report['session_options']}",
        "- Clean-path check: the PyTorch path reproduced the Milestone 4 test logits "
        f"bit-for-bit: {report['clean_path_check']['reproduced_baseline_test_logits']}",
        f"- Tolerances: {tol}",
        f"- **Parity passed: {report['passed']}**",
        "",
        "| Set | Windows | max abs logit diff | mean abs logit diff | max abs prob diff "
        "| max abs aggregate diff | top-1 flips (near-tie) | violating windows | passed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, part in (("A: clip streams", a), ("B: concatenation streams", b)):
        logits, agg = part["logits"], part["aggregates"]
        lines.append(
            f"| {name} | {logits['windows']} | {logits['abs_logit_diff']['max']:.3e} | "
            f"{logits['abs_logit_diff']['mean']:.3e} | "
            f"{logits['abs_probability_diff']['max']:.3e} | {agg['abs_diff']['max']:.3e} | "
            f"{logits['top1_disagreements']} ({logits['top1_near_tie_flips']}) | "
            f"{logits['violating_windows'] + agg['violating_windows']} | "
            f"{logits['passed'] and agg['passed']} |"
        )
    c = report["streaming_consistency"]
    lines += [
        "",
        "## Streaming consistency (set A, all test clips)",
        "",
        "| Path | Stream decision = Milestone 4 prediction | Accuracy |",
        "| --- | ---: | ---: |",
    ]
    for name in ("onnxruntime", "pytorch"):
        part = c[name]
        lines.append(
            f"| {name} streaming, batch 1 | {part['agree_with_milestone4']}/{c['clips']} "
            f"({part['agreement_rate']:.4f}) | {part['accuracy']:.4f} |"
        )
    lines += [
        f"| Milestone 4 clip-level (batch 256) | - | {c['milestone4_test_accuracy']:.4f} |",
        "",
        "Full statistics, violations (if any), and artifact hashes: `report.json`.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")
