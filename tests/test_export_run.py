"""Synthetic end-to-end export and parity run (kw10-onnx-v1); no network, no real dataset."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from robust_streaming_ml.config import load_config, parse_config
from robust_streaming_ml.data.manifest import read_manifest
from robust_streaming_ml.evaluation.metrics import softmax
from robust_streaming_ml.evaluation.predictions import infer_logits
from robust_streaming_ml.evaluation.run import evaluate
from robust_streaming_ml.export import onnx_model
from robust_streaming_ml.export import run as export_run
from robust_streaming_ml.export.config import ExportConfig
from robust_streaming_ml.export.onnx_model import (
    OnnxModelError,
    OnnxWindowBackend,
    export_onnx_bytes,
    installed_versions,
    load_onnx_model,
)
from robust_streaming_ml.export.parity import causal_aggregates, compare_aggregates, compare_logits
from robust_streaming_ml.export.run import (
    ExportError,
    ExportResult,
    chunk_sizes,
    export_and_verify,
    load_verified_export,
    split_chunks,
)
from robust_streaming_ml.robustness.config import RobustnessConfig
from robust_streaming_ml.run_dir import ProvenanceError
from tests.audio_fixtures import tone, wav_bytes
from tests.evaluation_fixtures import Trained, apply_overrides, make_trained
from tests.training_fixtures import training_files

LABELS = ("yes", "no")


def files_with_long_test_clips() -> dict[str, bytes]:
    """Toy files whose two test clips (16,000 and 12,000 frames) form a 9-window stream."""
    files = training_files()
    files["yes/dddddddd_nohash_0.wav"] = wav_bytes(tone(16000, 71))
    files["no/dddddddd_nohash_0.wav"] = wav_bytes(tone(12000, 72))
    return files


@dataclass
class Setup:
    trained: Trained
    baseline_dir: Path

    @property
    def tmp(self) -> Path:
        return self.trained.tmp

    def raw(self) -> dict[str, Any]:
        evaluation = self.trained.raw()
        return {
            "schema_version": 1,
            "name": "toy-onnx",
            "seed": 7,
            "deterministic": True,
            "run": {"output_root": str(self.tmp / "export_runs")},
            "data": evaluation["data"],
            "checkpoint": evaluation["checkpoint"],
            "baseline": {
                "run_dir": str(self.baseline_dir),
                "report_sha256": hashlib.sha256(
                    (self.baseline_dir / "report.json").read_bytes()
                ).hexdigest(),
            },
            "export": {"reference_batch_size": 3, "num_threads": 1},
        }

    def config(self, **overrides: Any) -> ExportConfig:
        return parse_config(apply_overrides(self.raw(), overrides), schema=ExportConfig)

    def run(self, **overrides: Any) -> ExportResult:
        return export_and_verify(self.config(**overrides), dataset_spec=self.trained.toy.spec)

    def runs(self) -> list[Path]:
        root = self.tmp / "export_runs"
        return sorted(root.iterdir()) if root.exists() else []


@pytest.fixture
def installed_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the protocol versions to the installed ones (CI locks another onnxruntime)."""
    found = installed_versions()
    monkeypatch.setattr(
        onnx_model,
        "EXPORT_TOOLCHAIN",
        {k: str(found[k]) for k in ("torch", "onnx", "onnxscript")},
    )
    monkeypatch.setattr(onnx_model, "RUNTIME_VERSION", str(found["onnxruntime"]))


@pytest.fixture
def setup(tmp_path: Path, package_repo: Path, installed_pins: None) -> Setup:
    trained = make_trained(tmp_path, files_with_long_test_clips())
    baseline = evaluate(trained.config(), dataset_spec=trained.toy.spec)
    return Setup(trained=trained, baseline_dir=baseline.run.path)


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def logits_of(rows: list[dict[str, str]], prefix: str) -> npt.NDArray[np.float32]:
    return np.asarray(
        [[float(r[f"{prefix}_logit_{label}"]) for label in LABELS] for r in rows],
        dtype=np.float32,
    )


# --- end to end -------------------------------------------------------------------------


def test_end_to_end_export_and_parity(setup: Setup) -> None:
    manifest = setup.trained.toy.built.manifest_path
    before = {r.relative_path: r.sha256 for r in read_manifest(manifest) if r.split == "test"}
    result = setup.run()
    run = result.run.path
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report == result.report and result.passed and report["passed"] is True
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["parity_passed"] is True
    for name, digest in report["artifacts_sha256"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    assert report["clean_path_check"]["reproduced_baseline_test_logits"] is True
    assert report["onnx_model"]["deterministic_export"] is True
    assert report["protocols"]["streaming"]["hop_samples"] == 1600
    assert report["session_options"]["intra_op_num_threads"] == 1

    # the model is tied to the checkpoint, labels, and source revision
    model, verified = load_verified_export(
        run, report_sha256=hashlib.sha256((run / "report.json").read_bytes()).hexdigest()
    )
    assert verified == report and model.labels == LABELS
    assert model.metadata["checkpoint"]["sha256"] == setup.raw()["checkpoint"]["sha256"]
    assert model.metadata["source"] == result.run.metadata["source"]

    # set A: every statistic regenerates from clips.csv
    clips = read_csv(run / "parity" / "clips.csv")
    assert [c["relative_path"] for c in clips] == sorted(before)
    a = report["parity"]["clip_streams"]
    regenerated = compare_logits(
        logits_of(clips, "torch"), logits_of(clips, "onnx"), row_ids=sorted(before)
    )
    assert regenerated == a["logits"]
    consistency = report["streaming_consistency"]
    onnx_decisions = [int(c["onnx_decision_index"]) for c in clips]
    m4 = [int(c["milestone4_pred_index"]) for c in clips]
    agree = sum(o == m for o, m in zip(onnx_decisions, m4, strict=True))
    assert consistency["onnxruntime"]["agree_with_milestone4"] == agree
    assert consistency["clips"] == 2 and [int(c["valid_samples"]) for c in clips] == [12000, 16000]

    # set B: one 28,000-sample stream -> 8 full windows and one final partial window
    windows = read_csv(run / "parity" / "stream_windows.csv")
    assert [int(w["start_sample"]) for w in windows] == [1600 * k for k in range(9)]
    assert [w["final_partial"] for w in windows] == ["false"] * 8 + ["true"]
    assert int(windows[-1]["valid_samples"]) == 28000 - 12800
    b = report["parity"]["concatenation_streams"]
    torch_logits, onnx_logits = logits_of(windows, "torch"), logits_of(windows, "onnx")
    ids = [f"stream-0/{k}" for k in range(9)]
    assert compare_logits(torch_logits, onnx_logits, row_ids=ids) == b["logits"]
    aggregates = compare_aggregates(
        causal_aggregates(softmax(torch_logits), 5),
        causal_aggregates(softmax(onnx_logits), 5),
        row_ids=ids,
    )
    assert aggregates == b["aggregates"]
    (stream,) = read_csv(run / "parity" / "streams.csv")
    assert stream["num_samples"] == "28000" and stream["num_windows"] == "9"
    assert stream["relative_paths"] == "|".join(sorted(before))

    root = setup.trained.toy.toy.root  # source audio unchanged
    for path, digest in before.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


def test_a_rerun_exports_identical_model_bytes(setup: Setup) -> None:
    first = setup.run().run.path / "onnx" / "model.onnx"
    second = setup.run().run.path / "onnx" / "model.onnx"
    assert first.read_bytes() == second.read_bytes()


def test_repository_config_pins_the_evaluated_checkpoint_and_baseline() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(configs / "kw10_cnn_baseline_onnx.yaml", schema=ExportConfig)
    robust = load_config(configs / "kw10_cnn_baseline_robustness.yaml", schema=RobustnessConfig)
    assert config.checkpoint == robust.checkpoint and config.baseline == robust.baseline
    assert config.data == robust.data
    assert (config.export.reference_batch_size, config.export.num_threads) == (256, 8)


def test_chunk_sizes_are_seeded_log_uniform_and_cover_the_stream() -> None:
    sizes = chunk_sizes("stream-3", 100000)
    assert sum(sizes) == 100000 and min(sizes) >= 1 and max(sizes) <= 32000
    assert sizes == chunk_sizes("stream-3", 100000) != chunk_sizes("stream-4", 100000)
    signal = np.arange(10)
    assert [c.tolist() for c in split_chunks(signal, [3, 0, 7])] == [
        [0, 1, 2],
        [],
        [3, 4, 5, 6, 7, 8, 9],
    ]
    with pytest.raises(ExportError):
        split_chunks(signal, [3, 3])


# --- gates: nothing is written ----------------------------------------------------------


def assert_rejected(setup: Setup, error: type[Exception], match: str, **overrides: Any) -> None:
    with pytest.raises(error, match=match):
        setup.run(**overrides)
    assert setup.runs() == []


def test_unpinned_baseline_is_refused(setup: Setup) -> None:
    assert_rejected(setup, ExportError, "pinned", **{"baseline.report_sha256": "0" * 64})


def test_reference_batch_size_must_match_the_baseline(setup: Setup) -> None:
    assert_rejected(setup, ExportError, "batch_size", **{"export.reference_batch_size": 4})


def test_other_export_toolchain_is_refused(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onnx_model, "EXPORT_TOOLCHAIN", {"onnxscript": "0.0.0"})
    assert_rejected(setup, OnnxModelError, "export toolchain differs")


def test_other_runtime_version_is_refused(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onnx_model, "RUNTIME_VERSION", "0.0.0")
    assert_rejected(setup, OnnxModelError, "pins 0.0.0")


def test_dirty_source_is_refused(setup: Setup, package_repo: Path) -> None:
    (package_repo / "wip.py").write_text("x = 1\n", encoding="utf-8")
    assert_rejected(setup, ProvenanceError, "uncommitted")


def test_cli_rejects_an_unpinned_baseline(setup: Setup) -> None:
    path = setup.tmp / "onnx.yaml"
    raw = setup.raw()
    raw["baseline"]["report_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_onnx.py"
    main = runpy.run_path(str(script), run_name="export_onnx")["main"]
    assert main(["--config", str(path)]) == 1
    assert setup.runs() == []


# --- failures inside the run --------------------------------------------------------------


def test_parity_violation_is_reported_and_fails_the_run(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = OnnxWindowBackend.__call__

    def perturbed(self: OnnxWindowBackend, windows: Any) -> Any:
        out = original(self, windows)
        out[:, 0] += np.float32(0.05)
        return out

    monkeypatch.setattr(OnnxWindowBackend, "__call__", perturbed)
    result = setup.run()
    run = result.run.path
    assert not result.passed
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and "tolerances violated" in summary["error"]
    a = result.report["parity"]["clip_streams"]["logits"]
    assert a["violations_by_kind"]["logit"] == 2 and a["violations"][0]["kinds"][0] == "logit"
    assert (run / "parity" / "clips.csv").is_file() and (run / "report.md").is_file()
    digest = hashlib.sha256((run / "report.json").read_bytes()).hexdigest()
    with pytest.raises(ExportError, match="did not pass parity"):
        load_verified_export(run, report_sha256=digest)


def test_clean_path_mismatch_fails_the_run(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    original = infer_logits

    def shifted(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs) + np.float32(1e-3)

    monkeypatch.setattr(export_run, "infer_logits", shifted)
    with pytest.raises(ExportError, match="does not reproduce"):
        setup.run()
    (run,) = setup.runs()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and not (run / "report.json").exists()


def test_non_deterministic_export_fails_the_run(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = export_onnx_bytes
    calls: list[int] = []

    def varying(*args: Any, **kwargs: Any) -> bytes:
        calls.append(1)
        data = original(*args, **kwargs)
        return data if len(calls) == 1 else data + b"\x00"

    monkeypatch.setattr(export_run, "export_onnx_bytes", varying)
    with pytest.raises(OnnxModelError, match="different bytes"):
        setup.run()
    (run,) = setup.runs()
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_verified_export_rejects_wrong_pins_and_tampering(setup: Setup) -> None:
    run = setup.run().run.path
    digest = hashlib.sha256((run / "report.json").read_bytes()).hexdigest()
    with pytest.raises(ExportError, match="pinned"):
        load_verified_export(run, report_sha256="0" * 64)
    with pytest.raises(OnnxModelError, match="labels"):
        load_verified_export(run, report_sha256=digest, expect_labels=["no", "yes"])
    model_path = run / "onnx" / "model.onnx"
    model_path.write_bytes(model_path.read_bytes() + b"\x00")
    with pytest.raises(OnnxModelError, match="does not match sidecar"):
        load_verified_export(run, report_sha256=digest)
    with pytest.raises(OnnxModelError):
        load_onnx_model(model_path)
