"""CPU latency benchmark kw10-latency-v1: statistics, environment, and a synthetic run."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robust_streaming_ml.benchmark.config import BenchmarkConfig
from robust_streaming_ml.benchmark.hardware import cpu_info, environment_record, parse_cpuinfo
from robust_streaming_ml.benchmark.run import (
    PATHS,
    PROTOCOL_PLAN,
    BenchmarkError,
    BenchmarkPlan,
    BenchmarkResult,
    run_benchmark,
)
from robust_streaming_ml.benchmark.stats import nearest_rank, summarize
from robust_streaming_ml.config import load_config, parse_config
from robust_streaming_ml.export import onnx_model
from robust_streaming_ml.export.onnx_model import OnnxModelError, OnnxWindowBackend
from robust_streaming_ml.export.run import ExportError
from robust_streaming_ml.run_dir import ProvenanceError
from tests.evaluation_fixtures import apply_overrides
from tests.export_fixtures import (
    ExportSetup,
    make_export_setup,
    pin_installed_versions,
    sha256_file,
)

SMALL = BenchmarkPlan(warmup=2, timed=3, repetitions=2, threads=(1,), sanity_windows=2)

# --- statistics ------------------------------------------------------------------------------


def test_nearest_rank_percentiles_by_hand() -> None:
    values = list(range(1, 11))  # x(1)..x(10)
    assert nearest_rank(values, 50) == 5  # ceil(0.5 * 10) = 5
    assert nearest_rank(values, 95) == 10  # ceil(9.5) = 10
    assert nearest_rank(values, 100) == 10 and nearest_rank(values, 1) == 1
    twenty = list(range(1, 21))
    assert nearest_rank(twenty, 95) == 19 and nearest_rank(twenty, 50) == 10
    pooled = list(range(15000))
    assert nearest_rank(pooled, 95) == 14249  # x(14250), 0-based value 14249
    assert nearest_rank(pooled, 50) == 7499
    assert nearest_rank([7], 50) == 7


def test_nearest_rank_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no values"):
        nearest_rank([], 50)
    with pytest.raises(ValueError, match="percent"):
        nearest_rank([1], 0)
    with pytest.raises(ValueError, match="sorted"):
        nearest_rank([2, 1], 50)


def test_summarize_in_milliseconds() -> None:
    summary = summarize([4_000_000, 1_000_000, 3_000_000, 2_000_000])
    assert summary == {
        "n": 4,
        "p50_ms": 2.0,
        "p95_ms": 4.0,
        "p99_ms": 4.0,
        "mean_ms": 2.5,
        "min_ms": 1.0,
        "max_ms": 4.0,
    }
    with pytest.raises(ValueError, match="no durations"):
        summarize([])
    with pytest.raises(ValueError, match="non-negative"):
        summarize([-1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timed": 0},
        {"repetitions": 0},
        {"warmup": -1},
        {"threads": ()},
        {"threads": (0,)},
        {"warmup": 1, "timed": 1, "sanity_windows": 3},
    ],
)
def test_invalid_plans_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkPlan(**kwargs)


def test_protocol_plan_is_fixed() -> None:
    assert (PROTOCOL_PLAN.warmup, PROTOCOL_PLAN.timed, PROTOCOL_PLAN.repetitions) == (500, 5000, 3)
    assert PROTOCOL_PLAN.threads == (1, 4)
    assert PATHS == ("onnx_streaming_step", "onnx_session_run", "pytorch_streaming_step")


# --- environment -------------------------------------------------------------------------------


def test_cpuinfo_parser_counts_physical_cores() -> None:
    text = "\n".join(
        [
            "processor\t: 0\nmodel name\t: Example CPU 9000\ncpu MHz\t\t: 3200.000\n"
            "physical id\t: 0\ncore id\t\t: 0\n",
            "processor\t: 1\nmodel name\t: Example CPU 9000\nphysical id\t: 0\ncore id\t\t: 0\n",
            "processor\t: 2\nmodel name\t: Example CPU 9000\nphysical id\t: 0\ncore id\t\t: 1\n",
            "processor\t: 3\nmodel name\t: Example CPU 9000\nphysical id\t: 0\ncore id\t\t: 1",
        ]
    )
    info = parse_cpuinfo(text)
    assert info["model"] == "Example CPU 9000" and info["nominal_mhz"] == 3200.0
    assert info["physical_cores"] == 2
    assert parse_cpuinfo("processor : 0\n")["physical_cores"] is None


def test_cpu_info_and_environment_are_recorded() -> None:
    info = cpu_info()
    assert info["logical_cores"] == os.cpu_count()
    physical = info["physical_cores"]
    assert physical is None or 1 <= physical <= (os.cpu_count() or physical)
    record = environment_record()
    assert record["timer"]["function"] == "time.perf_counter_ns"
    assert record["packages"]["onnxruntime"] and record["python"]["version"]
    assert "CPUExecutionProvider" in record["onnxruntime_providers"]


# --- synthetic end-to-end ------------------------------------------------------------------


@dataclass
class BenchSetup:
    export: ExportSetup
    export_dir: Path

    @property
    def tmp(self) -> Path:
        return self.export.tmp

    def raw(self) -> dict[str, Any]:
        base = self.export.raw()
        return {
            "schema_version": 1,
            "name": "toy-bench",
            "seed": 7,
            "deterministic": True,
            "run": {"output_root": str(self.tmp / "bench_runs")},
            "data": base["data"],
            "checkpoint": base["checkpoint"],
            "export": {
                "run_dir": str(self.export_dir),
                "report_sha256": sha256_file(self.export_dir / "report.json"),
            },
        }

    def config(self, **overrides: Any) -> BenchmarkConfig:
        return parse_config(apply_overrides(self.raw(), overrides), schema=BenchmarkConfig)

    def run(self, plan: BenchmarkPlan = SMALL, **overrides: Any) -> BenchmarkResult:
        return run_benchmark(
            self.config(**overrides), dataset_spec=self.export.trained.toy.spec, plan=plan
        )

    def runs(self) -> list[Path]:
        root = self.tmp / "bench_runs"
        return sorted(root.iterdir()) if root.exists() else []


@pytest.fixture
def bench(tmp_path: Path, package_repo: Path, monkeypatch: pytest.MonkeyPatch) -> BenchSetup:
    pin_installed_versions(monkeypatch)
    export = make_export_setup(tmp_path)
    result = export.run()
    assert result.passed
    return BenchSetup(export=export, export_dir=result.run.path)


def read_timings(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def test_end_to_end_benchmark(bench: BenchSetup) -> None:
    plan = BenchmarkPlan(warmup=2, timed=3, repetitions=2, threads=(1, 2), sanity_windows=2)
    result = bench.run(plan=plan)
    run = result.run.path
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report == result.report
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["plan_is_protocol"] is False
    for name, digest in summary["artifacts_sha256"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    rows = read_timings(run / "timings.csv")
    assert len(rows) == 3 * 2 * 2 * 3  # paths x threads x repetitions x timed steps
    assert {r["window_index"] for r in rows} == {"2", "3", "4"}  # warm-up windows 0-1 dropped
    order = list(dict.fromkeys((r["repetition"], r["threads"], r["path"]) for r in rows))
    assert order == [(rep, t, p) for rep in ("1", "2") for t in ("1", "2") for p in PATHS]
    for key, entry in report["results"].items():
        path, threads = key.split("/threads=")
        mine = [int(r["ns"]) for r in rows if r["path"] == path and r["threads"] == threads]
        assert entry["pooled"] == summarize(mine) and entry["pooled"]["n"] == 6
        assert len(entry["repetitions"]) == 2
    assert report["sanity_checks"]["streaming_equals_offline_onnx"] is True
    assert report["model"]["export_parity_passed"] is True
    assert report["audio"]["clips"] == 2 and report["audio"]["samples"] == 14400 + 5 * 1600
    assert report["environment"]["cpu"]["logical_cores"] == os.cpu_count()
    assert report["session_options"]["2"]["intra_op_num_threads"] == 2
    assert "not a production SLA" in (run / "report.md").read_text(encoding="utf-8")


def test_the_plan_must_fit_the_available_audio(bench: BenchSetup) -> None:
    with pytest.raises(BenchmarkError, match="the plan needs"):
        bench.run(plan=BenchmarkPlan(warmup=2, timed=100, repetitions=1, threads=(1,)))
    (run,) = bench.runs()
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_sanity_check_failure_stops_the_run(
    bench: BenchSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = OnnxWindowBackend.__call__

    def perturbed(self: OnnxWindowBackend, windows: Any) -> Any:
        out = original(self, windows)
        return out + np.float32(0.5)

    monkeypatch.setattr(OnnxWindowBackend, "__call__", perturbed)
    with pytest.raises(BenchmarkError, match="sanity check failed"):
        bench.run()
    (run,) = bench.runs()
    assert not (run / "timings.csv").exists()


# --- gates: nothing is written ----------------------------------------------------------


def assert_rejected(
    bench: BenchSetup, error: type[Exception], match: str, **overrides: Any
) -> None:
    with pytest.raises(error, match=match):
        bench.run(**overrides)
    assert bench.runs() == []


def test_unpinned_export_report_is_refused(bench: BenchSetup) -> None:
    assert_rejected(bench, ExportError, "pinned", **{"export.report_sha256": "0" * 64})


def test_an_export_that_failed_parity_is_refused(bench: BenchSetup) -> None:
    report_path = bench.export_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert_rejected(
        bench,
        ExportError,
        "did not pass parity",
        **{"export.report_sha256": sha256_file(report_path)},
    )


def test_other_runtime_version_is_refused(
    bench: BenchSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(onnx_model, "RUNTIME_VERSION", "0.0.0")
    assert_rejected(bench, OnnxModelError, "pins 0.0.0")


def test_dirty_source_is_refused(bench: BenchSetup, package_repo: Path) -> None:
    (package_repo / "wip.py").write_text("x = 1\n", encoding="utf-8")
    assert_rejected(bench, ProvenanceError, "uncommitted")


def test_cli_rejects_an_unpinned_export(bench: BenchSetup) -> None:
    path = bench.tmp / "bench.yaml"
    raw = bench.raw()
    raw["export"]["report_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_latency.py"
    main = runpy.run_path(str(script), run_name="benchmark_latency")["main"]
    assert main(["--config", str(path)]) == 1
    assert bench.runs() == []


def test_repository_config_pins_the_milestone6_export() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(configs / "kw10_onnx_benchmark.yaml", schema=BenchmarkConfig)
    assert config.export.run_dir == Path("runs/20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b")
    assert config.export.report_sha256.startswith("7fddcac2")
    assert config.checkpoint.sha256.startswith("d21f8928")
