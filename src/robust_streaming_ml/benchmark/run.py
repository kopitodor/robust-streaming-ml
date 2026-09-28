"""CPU latency benchmark entry point (protocol ``kw10-latency-v1``).

Before anything is written, :func:`run_benchmark` checks:

1. the pinned checkpoint and its provenance (the Milestone 4 gate);
2. the validated manifest;
3. the pinned Milestone 6 export: report SHA-256, passed parity, and the
   model's checkpoint, labels, features, manifest, and file hashes;
4. ONNX Runtime at the pinned version with the CPU provider;
5. a clean source tree.

It then calls ``start_run``. Inside the run it builds one stream from the
official test clips in manifest order, checks the ONNX streaming path on the
first windows (bit-identical to offline ONNX Runtime, and within the
``kw10-onnx-v1`` tolerances of PyTorch), and times every measurement block:
``for repetition: for threads in (1, 4): for path``. Each block has a fresh
session or classifier, 500 untimed warm-up steps, and 5,000 individually
timed steps (``time.perf_counter_ns``). It writes the raw ``timings.csv``,
``report.json``, ``report.md``, and ``summary.json``.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from robust_streaming_ml.benchmark.config import BenchmarkConfig
from robust_streaming_ml.benchmark.hardware import environment_record
from robust_streaming_ml.benchmark.stats import summarize
from robust_streaming_ml.checkpoint import LoadedCheckpoint
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import DatasetSpec
from robust_streaming_ml.evaluation.run import verify_checkpoint
from robust_streaming_ml.export.onnx_model import (
    INPUT_NAME,
    OUTPUT_NAME,
    LoadedOnnxModel,
    OnnxWindowBackend,
    create_session,
    require_runtime,
    session_options_record,
)
from robust_streaming_ml.export.parity import compare_logits
from robust_streaming_ml.export.run import load_verified_export
from robust_streaming_ml.run_dir import (
    ProvenanceError,
    RunDirectory,
    current_source_revision,
    start_run,
)
from robust_streaming_ml.streaming.classifier import (
    StreamingClassifier,
    TorchWindowBackend,
    WindowBackend,
)
from robust_streaming_ml.streaming.windowing import StreamingSpec, stream_samples
from robust_streaming_ml.training.config import WindowSection
from robust_streaming_ml.training.data import (
    ValidatedManifest,
    check_manifest_inputs,
    read_split_audio,
)

__all__ = [
    "LATENCY_PROTOCOL_ID",
    "PATHS",
    "PROTOCOL_PLAN",
    "BenchmarkError",
    "BenchmarkPlan",
    "BenchmarkResult",
    "run_benchmark",
]

LATENCY_PROTOCOL_ID = "kw10-latency-v1"
PATHS = ("onnx_streaming_step", "onnx_session_run", "pytorch_streaming_step")
PATH_DEFINITIONS = {
    "onnx_streaming_step": "StreamingClassifier.push(1,600 int16 samples) completing one window "
    "on ONNX Runtime: chunk validation and int16->float32, buffer update, window copy, "
    "inference including the log-mel front end (STFT, mel, log) inside the graph, float64 "
    "softmax, argmax, causal aggregate. Excludes file I/O, decoding, model loading.",
    "onnx_session_run": "InferenceSession.run on one pre-extracted [1, 16000] float32 window "
    "(front end + CNN inside the graph), without streaming bookkeeping.",
    "pytorch_streaming_step": "The same streaming step with the PyTorch eager backend "
    "(validated front end + CNN, no_grad); an unoptimised reference, not a tuned deployment.",
}
SPLIT = "test"
REPORT_SCHEMA_VERSION = 1

Int16 = npt.NDArray[np.int16]
Float32 = npt.NDArray[np.float32]


class BenchmarkError(RuntimeError):
    """A gate or sanity check failed, or the benchmark cannot run as specified."""


@dataclass(frozen=True)
class BenchmarkPlan:
    """Block sizes; the protocol fixes :data:`PROTOCOL_PLAN` (tests use smaller plans)."""

    warmup: int = 500
    timed: int = 5000
    repetitions: int = 3
    threads: tuple[int, ...] = (1, 4)
    sanity_windows: int = 20

    def __post_init__(self) -> None:
        if min(self.timed, self.repetitions, self.sanity_windows) < 1 or self.warmup < 0:
            raise BenchmarkError("plan sizes must be positive (warm-up may be 0)")
        if not self.threads or min(self.threads) < 1:
            raise BenchmarkError("plan needs at least one positive thread count")
        if self.sanity_windows > self.warmup + self.timed:
            raise BenchmarkError("sanity windows must lie within the benchmark stream")


PROTOCOL_PLAN = BenchmarkPlan()


@dataclass(frozen=True)
class BenchmarkResult:
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


def run_benchmark(
    config: BenchmarkConfig,
    *,
    dataset_spec: DatasetSpec = SPEECH_COMMANDS_KW10,
    root: Path | None = None,
    run_id: str | None = None,
    allow_dirty_source: bool = False,
    plan: BenchmarkPlan = PROTOCOL_PLAN,
) -> BenchmarkResult:
    """Run the protocol; see the module docstring. ``plan`` exists for tests only."""
    loaded = verify_checkpoint(config.checkpoint, config.data)
    window = WindowSection.model_validate(loaded.metadata["preprocessing"]["window"])
    manifest = check_manifest_inputs(config.data, window, dataset_spec)
    model, export_report = load_verified_export(
        config.export.run_dir,
        report_sha256=config.export.report_sha256,
        expect_checkpoint_sha256=loaded.sha256,
        expect_labels=manifest.labels,
    )
    runtime = require_runtime()
    source = current_source_revision()
    if source.dirty and not allow_dirty_source:
        raise ProvenanceError(
            f"working tree has uncommitted changes at {source.commit}; commit them "
            "(or pass allow_dirty_source for a non-result development run)"
        )
    run = start_run(config, root=root, run_id=run_id)
    try:
        report, summary = _run(
            config, run, loaded, manifest, window, model, export_report, runtime, plan
        )
    except BaseException as exc:
        _write_atomic(
            run.path / "summary.json",
            _json_bytes(
                {"status": "failed", "run_id": run.run_id, "error": f"{type(exc).__name__}: {exc}"}
            ),
        )
        raise
    return BenchmarkResult(run=run, report=report, summary=summary)


def _time_streaming(
    backend: WindowBackend,
    spec: StreamingSpec,
    labels: tuple[str, ...],
    stream: Int16,
    plan: BenchmarkPlan,
) -> list[int]:
    classifier = StreamingClassifier(backend, spec, labels, sample_rate_hz=spec.sample_rate_hz)
    size, hop = spec.window_samples, spec.hop_samples
    if classifier.push(stream[: size - hop]):
        raise BenchmarkError("pre-fill emitted a window")
    clock = time.perf_counter_ns
    durations: list[int] = []
    for step in range(plan.warmup + plan.timed):
        start = size - hop + step * hop
        chunk = stream[start : start + hop]
        began = clock()
        emitted = classifier.push(chunk)
        ended = clock()
        if len(emitted) != 1 or emitted[0].window.index != step:
            raise BenchmarkError(f"streaming step {step} did not emit exactly window {step}")
        if step >= plan.warmup:
            durations.append(ended - began)
    return durations


def _time_session(
    session: Any, stream: Float32, spec: StreamingSpec, num_classes: int, plan: BenchmarkPlan
) -> list[int]:
    size, hop = spec.window_samples, spec.hop_samples
    run: Callable[..., Any] = session.run
    outputs = [OUTPUT_NAME]
    clock = time.perf_counter_ns
    durations: list[int] = []
    for step in range(plan.warmup + plan.timed):
        feeds = {INPUT_NAME: stream[step * hop : step * hop + size].reshape(1, size)}
        began = clock()
        (logits,) = run(outputs, feeds)
        ended = clock()
        if logits.shape != (1, num_classes):
            raise BenchmarkError(f"session returned shape {logits.shape}")
        if step >= plan.warmup:
            durations.append(ended - began)
    return durations


def _sanity(
    model: LoadedOnnxModel,
    loaded: LoadedCheckpoint,
    spec: StreamingSpec,
    stream: Int16,
    stream_f32: Float32,
    plan: BenchmarkPlan,
) -> dict[str, Any]:
    size, hop, count = spec.window_samples, spec.hop_samples, plan.sanity_windows
    backend = OnnxWindowBackend(create_session(model, intra_op_threads=1))
    classifier = StreamingClassifier(
        backend, spec, model.labels, sample_rate_hz=spec.sample_rate_hz
    )
    streamed = classifier.push(stream[: size - hop + count * hop])
    windows = [stream_f32[k * hop : k * hop + size].reshape(1, size) for k in range(count)]
    offline = np.stack([backend(w)[0] for w in windows])
    torch_backend = TorchWindowBackend(loaded.model, loaded.frontend)
    reference = np.stack([torch_backend(w)[0] for w in windows])
    streamed_logits = np.stack([p.logits for p in streamed]) if streamed else offline[:0]
    identical = len(streamed) == count and np.array_equal(streamed_logits, offline)
    parity = compare_logits(reference, offline)
    if not identical or not parity["passed"]:
        raise BenchmarkError(
            f"sanity check failed: streaming==offline {identical}, parity {parity['passed']}"
        )
    return {
        "windows": count,
        "streaming_equals_offline_onnx": True,
        "onnx_vs_pytorch_passed": True,
        "onnx_vs_pytorch_max_abs_logit_diff": parity["abs_logit_diff"]["max"],
    }


def _run(
    config: BenchmarkConfig,
    run: RunDirectory,
    loaded: LoadedCheckpoint,
    manifest: ValidatedManifest,
    window: WindowSection,
    model: LoadedOnnxModel,
    export_report: dict[str, Any],
    runtime: dict[str, Any],
    plan: BenchmarkPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    environment = environment_record()  # before any block changes torch's thread count
    spec = StreamingSpec.from_window(model.window)
    labels = model.labels
    size, hop = spec.window_samples, spec.hop_samples
    needed = size - hop + (plan.warmup + plan.timed) * hop

    rows = manifest.rows_for(SPLIT)
    audio = read_split_audio(manifest, SPLIT, config.data, window)
    clips: list[Int16] = []
    total = 0
    for i, row in enumerate(rows):
        if total >= needed:
            break
        clips.append(audio.waveforms[i, : row.num_frames].numpy())
        total += row.num_frames
    if total < needed:
        raise BenchmarkError(f"the test split has {total} samples; the plan needs {needed}")
    stream: Int16 = np.concatenate(clips)[:needed]
    stream_f32 = stream_samples(stream)
    audio_record = {
        "source": "official test clips concatenated in manifest order",
        "clips": len(clips),
        "first_relative_path": rows[0].relative_path,
        "last_relative_path": rows[len(clips) - 1].relative_path,
        "samples": needed,
        "seconds": needed / spec.sample_rate_hz,
        "windows_per_block": plan.warmup + plan.timed,
    }
    sanity = _sanity(model, loaded, spec, stream, stream_f32, plan)

    rows_out: list[list[Any]] = []
    by_config: dict[str, list[list[int]]] = {}
    torch_threads = torch.get_num_threads()
    for repetition in range(1, plan.repetitions + 1):
        for threads in plan.threads:
            for path in PATHS:
                if path == "onnx_session_run":
                    session = create_session(model, intra_op_threads=threads)
                    durations = _time_session(session, stream_f32, spec, len(labels), plan)
                elif path == "onnx_streaming_step":
                    backend = OnnxWindowBackend(create_session(model, intra_op_threads=threads))
                    durations = _time_streaming(backend, spec, labels, stream, plan)
                else:
                    torch.set_num_threads(threads)
                    torch_backend = TorchWindowBackend(loaded.model, loaded.frontend)
                    durations = _time_streaming(torch_backend, spec, labels, stream, plan)
                by_config.setdefault(f"{path}/threads={threads}", []).append(durations)
                rows_out.extend(
                    [path, threads, repetition, step, plan.warmup + step, ns]
                    for step, ns in enumerate(durations)
                )
    torch.set_num_threads(torch_threads)

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["path", "threads", "repetition", "step", "window_index", "ns"])
    writer.writerows(rows_out)
    timings_sha = _write_atomic(run.path / "timings.csv", buffer.getvalue().encode("utf-8"))

    results: dict[str, Any] = {}
    for key, blocks in by_config.items():
        path, threads_text = key.split("/threads=")
        results[key] = {
            "path": path,
            "threads": int(threads_text),
            "pooled": summarize([ns for block in blocks for ns in block]),
            "repetitions": [summarize(block) for block in blocks],
        }
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "protocol": {
            "id": LATENCY_PROTOCOL_ID,
            "paths": list(PATHS),
            "path_definitions": PATH_DEFINITIONS,
            "plan": {**dataclasses.asdict(plan), "threads": list(plan.threads)},
            "plan_is_protocol": plan == PROTOCOL_PLAN,
            "order": "for repetition: for threads: for path (fresh session/classifier per block)",
            "batch_size": 1,
            "percentiles": "nearest rank x(ceil(q*N/100)) over integer nanoseconds; pooled over "
            "repetitions is primary",
            "streaming": spec.describe(),
            "process_policy": {
                "cpu_affinity": "not set",
                "priority": "normal (not changed)",
                "python_gc": "enabled",
                "power_plan": "not controlled",
                "background_load": "not controlled",
            },
        },
        "benchmark_run_id": run.run_id,
        "source": run.metadata["source"],
        "model": {
            "onnx_sha256": model.sha256,
            "export_run_id": export_report["export_run_id"],
            "export_report_sha256": config.export.report_sha256,
            "export_parity_passed": export_report["passed"],
            "checkpoint_sha256": loaded.sha256,
            "opset": model.metadata["contract"]["opset"],
            "labels": list(labels),
        },
        "runtime": runtime,
        "session_options": {
            str(t): session_options_record(intra_op_threads=t) for t in plan.threads
        },
        "environment": environment,
        "audio": audio_record,
        "sanity_checks": sanity,
        "results": results,
        "artifacts_sha256": {"timings.csv": timings_sha},
    }
    artifacts = {"timings.csv": timings_sha}
    artifacts["report.json"] = _write_atomic(run.path / "report.json", _json_bytes(report))
    artifacts["report.md"] = _write_atomic(run.path / "report.md", _markdown(report))
    summary = {
        "status": "completed",
        "run_id": run.run_id,
        "protocol": LATENCY_PROTOCOL_ID,
        "plan_is_protocol": plan == PROTOCOL_PLAN,
        "headline_ms": {
            key: {"p50": r["pooled"]["p50_ms"], "p95": r["pooled"]["p95_ms"]}
            for key, r in results.items()
        },
        "artifacts_sha256": artifacts,
        "seconds": round(time.perf_counter() - started, 3),
    }
    _write_atomic(run.path / "summary.json", _json_bytes(summary))
    return report, summary


def _markdown(report: dict[str, Any]) -> bytes:
    env = report["environment"]
    cpu = env["cpu"]
    plan = report["protocol"]["plan"]
    lines = [
        f"# Latency report: {report['protocol']['id']}",
        "",
        f"- Run `{report['benchmark_run_id']}` at `{report['source']['commit']}` "
        f"(dirty {report['source']['dirty']})",
        f"- ONNX model `{report['model']['onnx_sha256']}` (export run "
        f"`{report['model']['export_run_id']}`, parity passed "
        f"{report['model']['export_parity_passed']}); checkpoint "
        f"`{report['model']['checkpoint_sha256']}`",
        f"- Runtime: onnxruntime {report['runtime']['onnxruntime']} "
        f"({report['runtime']['provider']}); torch {env['packages']['torch']}; "
        f"Python {env['python']['version']}",
        f"- Machine: {cpu['model']} ({cpu['logical_cores']} logical, {cpu['physical_cores']} "
        f"physical cores); {env['os']['platform']}",
        f"- Plan: batch 1, window 16000, hop 1600; warm-up {plan['warmup']}, timed "
        f"{plan['timed']} x {plan['repetitions']} repetitions; threads {plan['threads']} "
        f"(protocol plan: {report['protocol']['plan_is_protocol']})",
        f"- Timer: {env['timer']['function']} ({env['timer']['implementation']}, resolution "
        f"{env['timer']['resolution_s']} s); nearest-rank percentiles",
        "",
        "Local measurement on one uncontrolled machine; not a production SLA or a real-time claim.",
        "",
        "| Path | Threads | N | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms "
        "| per-repetition p50 / p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report["results"].values():
        p = result["pooled"]
        reps = "; ".join(f"{r['p50_ms']:.3f} / {r['p95_ms']:.3f}" for r in result["repetitions"])
        lines.append(
            f"| {result['path']} | {result['threads']} | {p['n']} | {p['p50_ms']:.3f} | "
            f"{p['p95_ms']:.3f} | {p['p99_ms']:.3f} | {p['mean_ms']:.3f} | {p['min_ms']:.3f} | "
            f"{p['max_ms']:.3f} | {reps} |"
        )
    lines += ["", "Raw per-step timings: `timings.csv`. Full metadata: `report.json`.", ""]
    return "\n".join(lines).encode("utf-8")
