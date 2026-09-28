"""Local demo: input contract, streaming outputs, corruption contrast, page content, CLI."""

from __future__ import annotations

import json
import re
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from robust_streaming_ml.config import load_config, parse_config
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.demo.app import DemoError, read_demo_wav, run_demo
from robust_streaming_ml.demo.config import DemoConfig
from robust_streaming_ml.export.onnx_model import OnnxModelError, OnnxWindowBackend, create_session
from robust_streaming_ml.export.run import ExportError, load_verified_export
from robust_streaming_ml.robustness.corruptions import corrupt, example_seed, get_corruption
from robust_streaming_ml.streaming.classifier import classify_offline
from robust_streaming_ml.streaming.windowing import StreamingSpec
from tests.audio_fixtures import tone, wav_bytes
from tests.evaluation_fixtures import apply_overrides
from tests.export_fixtures import (
    ExportSetup,
    make_export_setup,
    pin_installed_versions,
    sha256_file,
)


@dataclass
class DemoSetup:
    export: ExportSetup
    export_dir: Path

    @property
    def tmp(self) -> Path:
        return self.export.tmp

    def raw(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": "toy-demo",
            "seed": 7,
            "export": {
                "run_dir": str(self.export_dir),
                "report_sha256": sha256_file(self.export_dir / "report.json"),
            },
            "demo": {"output_dir": str(self.tmp / "demo_out"), "max_seconds": 5},
        }

    def config(self, **overrides: Any) -> DemoConfig:
        return parse_config(apply_overrides(self.raw(), overrides), schema=DemoConfig)

    def run(self, audio: Path, **kwargs: Any) -> Any:
        overrides = kwargs.pop("overrides", {})
        return run_demo(
            self.config(**overrides),
            audio,
            dataset_spec=self.export.trained.toy.spec,
            **kwargs,
        )

    def audio(self, name: str, samples: npt.NDArray[np.int16], **wav: Any) -> Path:
        path = self.tmp / name
        path.write_bytes(wav_bytes(samples, **wav))
        return path


@pytest.fixture
def demo(tmp_path: Path, package_repo: Path, monkeypatch: pytest.MonkeyPatch) -> DemoSetup:
    pin_installed_versions(monkeypatch)
    export = make_export_setup(tmp_path)
    result = export.run()
    assert result.passed
    return DemoSetup(export=export, export_dir=result.run.path)


def offline_reference(setup: DemoSetup, samples: npt.NDArray[Any]) -> Any:
    model, _ = load_verified_export(
        setup.export_dir, report_sha256=sha256_file(setup.export_dir / "report.json")
    )
    backend = OnnxWindowBackend(create_session(model))
    spec = StreamingSpec.from_window(model.window)
    return classify_offline(samples, backend, spec, model.labels)


# --- input contract ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wav", "match"),
    [
        ({"channels": 2}, "mono 16000 Hz"),
        ({"rate": 8000}, "8000 Hz"),
        ({"format_tag": 3, "bits": 32}, "FORMAT_TAG_3"),
    ],
)
def test_inputs_outside_the_model_contract_are_refused(
    tmp_path: Path, wav: dict[str, Any], match: str
) -> None:
    path = tmp_path / "x.wav"
    samples = tone(3200, 1) if "bits" not in wav else np.zeros(1600, dtype=np.int32)
    path.write_bytes(wav_bytes(samples if "bits" not in wav else samples.tobytes(), **wav))
    with pytest.raises(DemoError, match=match):
        read_demo_wav(path, sample_rate_hz=16000, max_seconds=5)


def test_empty_long_malformed_and_missing_files_are_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(wav_bytes(b""))
    with pytest.raises(DemoError, match="no audio"):
        read_demo_wav(empty, sample_rate_hz=16000, max_seconds=5)
    long = tmp_path / "long.wav"
    long.write_bytes(wav_bytes(tone(16001, 2)))
    with pytest.raises(DemoError, match="limit is 1 s"):
        read_demo_wav(long, sample_rate_hz=16000, max_seconds=1)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF....WAVEjunk")
    with pytest.raises(DemoError, match="not a valid WAV"):
        read_demo_wav(bad, sample_rate_hz=16000, max_seconds=5)
    with pytest.raises(DemoError, match="cannot read"):
        read_demo_wav(tmp_path / "absent.wav", sample_rate_hz=16000, max_seconds=5)


# --- demo runs ---------------------------------------------------------------------------


def no_external_resources(page: str) -> bool:
    return "<script" not in page and not re.search(r'(src|href)="(?!data:)', page)


def test_clean_demo_matches_the_offline_reference(demo: DemoSetup) -> None:
    samples = tone(27000, 5)
    audio = demo.audio("clip & 'one'.wav", samples)
    result = demo.run(audio, output=demo.tmp / "out" / "page.html")
    record = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert record == result.record and result.json_path == demo.tmp / "out" / "page.json"
    reference, decision = offline_reference(demo, samples)
    windows = record["runs"]["clean"]["windows"]
    assert len(windows) == len(reference) == 8 and windows[-1]["final_partial"] is True
    for got, want in zip(windows, reference, strict=True):
        assert got["probabilities"] == [float(v) for v in want.probabilities]
        assert got["aggregate_label"] == ("yes", "no")[want.aggregate_label_index]
        assert got["inference_ms"] is not None and got["inference_ms"] > 0
    assert decision is not None and record["runs"]["clean"]["decision"]["label"] == decision.label
    assert record["runs"]["corrupted"] is None and record["corruption"] is None
    assert record["model"]["export_parity_passed"] is True
    assert record["model"]["runtime"]["providers"] == ["CPUExecutionProvider"]
    assert any("no silence or unknown-word rejection" in item for item in record["limitations"])
    page = result.html_path.read_text(encoding="utf-8")
    assert no_external_resources(page)
    assert "not production-ready" in page and "not a measured result" in page
    assert record["model"]["onnx_sha256"] in page and "Toy attribution" in page
    assert "clip &amp; &#x27;one&#x27;.wav" in page and "clip & 'one'.wav" not in page
    assert page.count("data:audio/wav;base64,") == 1


@pytest.mark.parametrize(("name", "severity"), [("clipping", "at_0.5_peak"), ("noise", "snr_10db")])
def test_corrupted_contrast_uses_the_protocol_transform(
    demo: DemoSetup, name: str, severity: str
) -> None:
    samples = tone(20000, 9)
    audio = demo.audio("clip.wav", samples)
    result = demo.run(audio, corruption=name, severity=severity)
    record = result.record
    spec = get_corruption(name)
    sev = spec.severity(severity)
    identity = f"demo:{record['input']['audio_sha256']}"
    assert record["corruption"]["identity"] == identity
    assert record["corruption"]["seed"] == example_seed(spec, sev, identity)
    clip = corrupt(spec, sev, samples, identity)
    reference, _ = offline_reference(demo, clip.samples)
    corrupted = record["runs"]["corrupted"]["windows"]
    assert [w["probabilities"] for w in corrupted] == [
        [float(v) for v in p.probabilities] for p in reference
    ]
    comparison = record["comparison"]
    assert comparison["windows"] == len(corrupted) == 4
    assert 0 <= comparison["same_window_label"] <= 4
    page = result.html_path.read_text(encoding="utf-8")
    assert result.html_path.parent == demo.tmp / "demo_out"
    assert page.count("data:audio/wav;base64,") == 2 and no_external_resources(page)
    assert f"{name} / {severity}" in page


def test_audio_embedding_can_be_turned_off(demo: DemoSetup) -> None:
    audio = demo.audio("clip.wav", tone(8000, 3))
    result = demo.run(audio, overrides={"demo.embed_audio": False})
    assert "data:audio" not in result.html_path.read_text(encoding="utf-8")


def test_corruption_arguments_are_validated(demo: DemoSetup) -> None:
    audio = demo.audio("clip.wav", tone(8000, 3))
    with pytest.raises(DemoError, match="both"):
        demo.run(audio, corruption="noise")
    with pytest.raises(DemoError, match="no severity"):
        demo.run(audio, corruption="noise", severity="snr_99db")


def test_an_unpinned_or_failed_export_is_refused(demo: DemoSetup) -> None:
    audio = demo.audio("clip.wav", tone(8000, 3))
    with pytest.raises(ExportError, match="pinned"):
        demo.run(audio, overrides={"export.report_sha256": "0" * 64})
    report_path = demo.export_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ExportError, match="did not pass parity"):
        demo.run(audio, overrides={"export.report_sha256": sha256_file(report_path)})


def test_a_model_for_another_dataset_is_refused(demo: DemoSetup) -> None:
    audio = demo.audio("clip.wav", tone(8000, 3))
    with pytest.raises(OnnxModelError, match="labels"):
        run_demo(demo.config(), audio, dataset_spec=SPEECH_COMMANDS_KW10)


def test_cli_writes_the_page_and_fails_cleanly(
    demo: DemoSetup, capsys: pytest.CaptureFixture[str]
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo.py"
    main = runpy.run_path(str(script), run_name="demo")["main"]
    assert main(["--list-corruptions"]) == 0
    assert "codec: opus_level_0.9" in capsys.readouterr().out
    config = demo.tmp / "demo.yaml"
    config.write_text(json.dumps(demo.raw()), encoding="utf-8")
    missing = demo.tmp / "absent.wav"
    assert main(["--config", str(config), "--audio", str(missing)]) == 1
    assert "demo failed" in capsys.readouterr().err


def test_repository_config_pins_the_milestone6_export() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(configs / "kw10_demo.yaml", schema=DemoConfig)
    assert config.export.run_dir == Path("runs/20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b")
    assert config.export.report_sha256.startswith("7fddcac2")
    assert config.demo.output_dir == Path("artifacts/demo")
