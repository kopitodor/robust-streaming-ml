"""Synthetic end-to-end robustness suite (no network, no real dataset)."""

from __future__ import annotations

import hashlib
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from robust_streaming_ml.config import parse_config
from robust_streaming_ml.data.manifest import read_manifest
from robust_streaming_ml.data.wav import parse_wav
from robust_streaming_ml.evaluation.metrics import classification_metrics
from robust_streaming_ml.evaluation.predictions import decode_predictions, infer_logits
from robust_streaming_ml.evaluation.run import evaluate
from robust_streaming_ml.robustness import corruptions as corruptions_module
from robust_streaming_ml.robustness import run as robust_run
from robust_streaming_ml.robustness.config import RobustnessConfig
from robust_streaming_ml.robustness.corruptions import (
    CORRUPTIONS,
    CodecBackendError,
    CorruptionError,
    corrupt,
    output_sha256,
)
from robust_streaming_ml.robustness.run import (
    RobustnessError,
    degradation,
    evaluate_robustness,
    top_confusions,
)
from robust_streaming_ml.run_dir import ProvenanceError
from tests.evaluation_fixtures import Trained, apply_overrides, make_trained

LABELS = ("yes", "no")


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
            "name": "toy-robust",
            "seed": 7,
            "deterministic": True,
            "run": {"output_root": str(self.tmp / "robust_runs")},
            "data": evaluation["data"],
            "checkpoint": evaluation["checkpoint"],
            "baseline": {
                "run_dir": str(self.baseline_dir),
                "report_sha256": hashlib.sha256(
                    (self.baseline_dir / "report.json").read_bytes()
                ).hexdigest(),
            },
            "robustness": {"batch_size": 3, "num_threads": 1, "corruption_workers": 2},
        }

    def config(self, **overrides: Any) -> RobustnessConfig:
        return parse_config(apply_overrides(self.raw(), overrides), schema=RobustnessConfig)

    def run(self, **overrides: Any) -> robust_run.RobustnessResult:
        return evaluate_robustness(self.config(**overrides), dataset_spec=self.trained.toy.spec)

    def runs(self) -> list[Path]:
        root = self.tmp / "robust_runs"
        return sorted(root.iterdir()) if root.exists() else []


@pytest.fixture
def setup(tmp_path: Path, package_repo: Path) -> Setup:
    trained = make_trained(tmp_path)
    baseline = evaluate(trained.config(), dataset_spec=trained.toy.spec)
    return Setup(trained=trained, baseline_dir=baseline.run.path)


def source_clips(setup: Setup) -> dict[str, npt.NDArray[np.int16]]:
    rows = [r for r in read_manifest(setup.trained.toy.built.manifest_path) if r.split == "test"]
    root = setup.trained.toy.toy.root
    clips = {
        r.relative_path: parse_wav((root / r.relative_path).read_bytes()).samples_int16()
        for r in rows
    }
    assert len(clips) == 2
    return clips


# --- end to end -------------------------------------------------------------------------


def test_end_to_end_robustness_run(setup: Setup) -> None:
    rows_before = {
        r.relative_path: r.sha256
        for r in read_manifest(setup.trained.toy.built.manifest_path)
        if r.split == "test"
    }
    result = setup.run()
    run = result.run.path
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report == result.report
    assert report["clean_path_check"] == {"reproduced_baseline_test_logits": True, "examples": 2}
    assert report["codec_backend"]["libsndfile"] == "1.2.2"
    assert [(c["corruption"], c["severity"]) for c in report["conditions"]] == [
        (spec.name, sev.name) for spec in CORRUPTIONS for sev in spec.severities
    ]
    baseline_report = json.loads((setup.baseline_dir / "report.json").read_text(encoding="utf-8"))
    clean = {
        k: baseline_report["metrics"]["test"]["raw"][k]
        for k in ("accuracy", "macro_f1", "nll", "ece")
    }
    assert report["baseline"]["clean_raw"] == clean

    clips = source_clips(setup)
    for condition in report["conditions"]:
        folder = run / "conditions" / condition["corruption"] / condition["severity"]
        preds = decode_predictions(
            (folder / "predictions.csv").read_bytes(), split="test", labels=LABELS
        )
        regenerated = classification_metrics(preds.logits, preds.y_true, LABELS, n_bins=15)
        assert condition["metrics"] == regenerated
        assert condition["degradation"] == {k: clean[k] - regenerated[k] for k in clean}
        spec = corruptions_module.get_corruption(condition["corruption"])
        sev = spec.severity(condition["severity"])
        manifest_lines = (
            (folder / "corruption_manifest.csv").read_text(encoding="utf-8").splitlines()
        )
        header = manifest_lines[0].split(",")
        for line in manifest_lines[1:]:
            record = dict(zip(header, line.split(","), strict=True))
            expected = corrupt(spec, sev, clips[record["relative_path"]], record["relative_path"])
            assert int(record["seed"]) == expected.seed
            assert record["output_sha256"] == output_sha256(expected.samples)
            assert record["corruption"] == spec.name and record["severity"] == sev.name
    for name, digest in report["artifacts_sha256"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["conditions"] == 15
    assert "Degradation = clean - corrupted" in (run / "report.md").read_text(encoding="utf-8")
    root = setup.trained.toy.toy.root
    for path, digest in rows_before.items():  # source audio unchanged
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    assert not (run / "audio").exists()  # no retention unless requested


def test_outputs_do_not_depend_on_worker_count(setup: Setup) -> None:
    one = setup.run(**{"robustness.corruption_workers": 1}).run.path
    four = setup.run(**{"robustness.corruption_workers": 4}).run.path
    for spec in CORRUPTIONS:
        for sev in spec.severities:
            for name in ("predictions.csv", "corruption_manifest.csv"):
                rel = Path("conditions") / spec.name / sev.name / name
                assert (one / rel).read_bytes() == (four / rel).read_bytes(), rel


def test_retained_audio_is_readable_and_exact(setup: Setup) -> None:
    run = setup.run(**{"robustness.retain_audio_examples": 1}).run.path
    clips = source_clips(setup)
    first = sorted(clips)[0]
    for spec in CORRUPTIONS:
        for sev in spec.severities:
            wav = parse_wav(
                (run / "audio" / spec.name / sev.name / first.replace("/", "__")).read_bytes()
            )
            assert (wav.sample_rate_hz, wav.num_channels, wav.sample_format) == (
                16000,
                1,
                "FORMAT_TAG_3_32BIT",
            )
            assert np.array_equal(
                wav.samples_float32(), corrupt(spec, sev, clips[first], first).samples
            )


# --- gates: nothing is written ----------------------------------------------------------


def assert_rejected(
    setup: Setup, config: RobustnessConfig, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        evaluate_robustness(config, dataset_spec=setup.trained.toy.spec)
    assert setup.runs() == []


def test_unpinned_baseline_report_is_refused(setup: Setup) -> None:
    assert_rejected(
        setup, setup.config(**{"baseline.report_sha256": "0" * 64}), RobustnessError, "pinned"
    )


def test_baseline_for_another_checkpoint_is_refused(setup: Setup) -> None:
    path = setup.baseline_dir / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["checkpoint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    assert_rejected(setup, setup.config(), RobustnessError, "baseline checkpoint")


def test_batch_size_must_match_the_baseline(setup: Setup) -> None:
    assert_rejected(
        setup, setup.config(**{"robustness.batch_size": 4}), RobustnessError, "batch_size"
    )


def test_missing_codec_backend_fails_before_start(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> Any:
        raise CodecBackendError("libsndfile missing (test)")

    monkeypatch.setattr(corruptions_module, "require_codec_backend", unavailable)
    assert_rejected(setup, setup.config(), CodecBackendError, "libsndfile missing")


def test_dirty_source_is_refused(setup: Setup, package_repo: Path) -> None:
    (package_repo / "wip.py").write_text("x = 1\n", encoding="utf-8")
    assert_rejected(setup, setup.config(), ProvenanceError, "uncommitted")


def test_cli_rejects_an_unpinned_baseline(setup: Setup) -> None:
    path = setup.tmp / "robust.yaml"
    raw = setup.raw()
    raw["baseline"]["report_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_robustness.py"
    main = runpy.run_path(str(script), run_name="evaluate_robustness")["main"]
    assert main(["--config", str(path)]) == 1
    assert setup.runs() == []


# --- failures inside the run are recorded -----------------------------------------------


def test_clean_path_mismatch_fails_the_run(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    original = infer_logits

    def perturbed(*args: Any, **kwargs: Any) -> npt.NDArray[np.float32]:
        return original(*args, **kwargs) + np.float32(1e-3)

    monkeypatch.setattr(robust_run, "infer_logits", perturbed)
    with pytest.raises(RobustnessError, match="does not reproduce"):
        setup.run()
    (run_path,) = setup.runs()
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and not (run_path / "conditions").exists()


def test_a_failing_corruption_fails_the_whole_run(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = corrupt

    def failing(spec: Any, severity: Any, clean: Any, path: str) -> Any:
        if spec.name == "codec":
            raise CorruptionError("synthetic codec failure")
        return original(spec, severity, clean, path)

    monkeypatch.setattr(robust_run, "corrupt", failing)
    with pytest.raises(CorruptionError, match="synthetic codec failure"):
        setup.run()
    (run_path,) = setup.runs()
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and "codec" in summary["error"]
    assert not (run_path / "report.json").exists()


# --- aggregate helpers ------------------------------------------------------------------------


def test_degradation_is_clean_minus_corrupted() -> None:
    clean = {"accuracy": 0.9, "macro_f1": 0.8, "nll": 0.3, "ece": 0.02}
    corrupted = {"accuracy": 0.6, "macro_f1": 0.5, "nll": 1.3, "ece": 0.12, "examples": 5}
    got = degradation(clean, corrupted)
    assert got == pytest.approx({"accuracy": 0.3, "macro_f1": 0.3, "nll": -1.0, "ece": -0.1})


def test_top_confusions_orders_by_count_then_label_order() -> None:
    metrics = {
        "confusion_matrix": {
            "labels": ["a", "b", "c"],
            "rows_true_cols_pred": [[5, 2, 3], [3, 4, 0], [0, 1, 6]],
        }
    }
    assert top_confusions(metrics, 3) == [
        {"true": "a", "predicted": "c", "count": 3},
        {"true": "b", "predicted": "a", "count": 3},
        {"true": "a", "predicted": "b", "count": 2},
    ]
