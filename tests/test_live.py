"""Live-microphone mode: resampling, audio contract, ordering, reset, and the local HTTP API."""

from __future__ import annotations

import http.client
import json
import math
import re
import threading
from collections.abc import Iterator
from fractions import Fraction
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from robust_streaming_ml.export.onnx_model import (
    LoadedOnnxModel,
    OnnxWindowBackend,
    create_session,
    export_onnx_bytes,
    load_onnx_model,
    model_metadata,
    write_onnx_model,
)
from robust_streaming_ml.live.resample import (
    KAISER_BETA,
    SUPPORTED_INPUT_RATES,
    ZERO_CROSSINGS,
    ResampleError,
    StreamingResampler,
    quantize_pcm16,
    resample_all,
)
from robust_streaming_ml.live.server import LiveDemoServer, LiveServerError, check_loopback
from robust_streaming_ml.live.session import (
    AudioContractError,
    ChunkOrderError,
    LiveSession,
    SessionStateError,
    decode_chunk,
)
from robust_streaming_ml.streaming.classifier import classify_offline
from robust_streaming_ml.streaming.windowing import StreamingSpec
from tests.test_onnx_export import toy_checkpoint

Float32 = npt.NDArray[np.float32]


def tone_bursts(rate: int, seconds: float, seed: int = 0) -> Float32:
    """Deterministic speech-band test signal at ``rate`` (tones plus a little noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(rate * seconds)) / rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.3 * t)
    x = 0.3 * np.sin(2 * np.pi * 440 * t) * envelope + 0.1 * np.sin(2 * np.pi * 2500 * t)
    return (x + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


def chunked(signal: Float32, sizes: list[int]) -> list[Float32]:
    out, pos, i = [], 0, 0
    while pos < signal.size:
        size = sizes[i % len(sizes)]
        out.append(signal[pos : pos + size])
        pos += size
        i += 1
    return out


def f32le(samples: npt.NDArray[Any]) -> bytes:
    return np.asarray(samples, dtype="<f4").tobytes()


def untimed(result: Any) -> Any:
    """A session result without the measured inference times (which vary run to run)."""
    if isinstance(result, list):
        return [untimed(r) for r in result]
    if isinstance(result, dict):
        return {k: untimed(v) for k, v in result.items() if k != "inference_ms"}
    return result


# --- resampling ------------------------------------------------------------------------------


def test_quantization_clips_and_rounds_half_to_even() -> None:
    values = np.array([-2.0, -1.0, 0.5 / 32768, 1.5 / 32768, 2.5 / 32768, 1.0, 3.0])
    assert quantize_pcm16(values).tolist() == [-32768, -32768, 0, 2, 2, 32767, 32767]


@pytest.mark.parametrize("rate", SUPPORTED_INPUT_RATES)
def test_output_count_is_ceil_of_the_rate_ratio(rate: int) -> None:
    for n in (1, 2, 3, 441, 1000, rate // 3 + 7):
        out = resample_all(tone_bursts(rate, n / rate + 1e-9)[:n], rate)
        assert out.dtype == np.int16 and out.size == math.ceil(n * 16000 / rate)


@pytest.mark.parametrize("rate", [48000, 44100, 22050, 8000, 96000])
def test_any_chunking_is_bit_identical(rate: int) -> None:
    signal = tone_bursts(rate, 0.73, seed=rate)
    reference = resample_all(signal, rate)
    rng = np.random.default_rng(rate)
    for sizes in ([1], [signal.size], [4800], [int(s) for s in rng.integers(1, 3000, 50)]):
        resampler = StreamingResampler(rate)
        parts = [resampler.push(c) for c in chunked(signal, sizes)] + [resampler.flush()]
        assert np.array_equal(np.concatenate(parts), reference)


def reference_output(signal: Float32, rate: int, n: int) -> float:
    """y[n] from the documented formula, computed independently of the implementation."""
    ratio = Fraction(rate, 16000)
    position = n * ratio
    cutoff = min(1.0, 16000 / rate)
    half = ZERO_CROSSINGS / cutoff
    phase_start = math.floor(position)
    frac = float(position - phase_start)

    def g(t: float) -> float:
        u = t / half
        if abs(u) >= 1.0:
            return 0.0
        window = float(np.i0(KAISER_BETA * math.sqrt(1.0 - u * u)) / np.i0(KAISER_BETA))
        return cutoff * float(np.sinc(cutoff * t)) * window

    reach = math.ceil(half)
    taps = {d: g(frac + d) for d in range(-reach, reach + 1)}
    norm = sum(taps.values())
    total = 0.0
    for d, tap in taps.items():
        k = phase_start - d
        if 0 <= k < signal.size:
            total += float(signal[k]) * tap / norm
    return total


@pytest.mark.parametrize("rate", [48000, 44100, 8000])
def test_matches_the_documented_formula(rate: int) -> None:
    signal = tone_bursts(rate, 0.2, seed=3)
    out = resample_all(signal, rate)
    for n in (0, 1, 7, 160, 1234, out.size - 1):
        expected = reference_output(signal, rate, n) * 32768
        assert abs(out[n] - expected) <= 0.5 + 1e-6, (n, out[n], expected)


def test_identity_at_16khz_and_unit_dc_gain() -> None:
    signal = tone_bursts(16000, 0.3)
    assert np.array_equal(resample_all(signal, 16000), quantize_pcm16(signal.astype(np.float64)))
    constant = np.full(48000, 0.25, dtype=np.float32)
    out = resample_all(constant, 48000)
    assert np.all(out[100:-100] == 8192)  # interior: exactly 0.25 after unit-DC-gain taps


def test_lowpass_keeps_speech_band_and_removes_aliases() -> None:
    t = np.arange(48000) / 48000
    passband = resample_all((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), 48000)
    stopband = resample_all((0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32), 48000)
    inner = slice(1000, -1000)
    peak = np.abs(passband[inner].astype(np.float64)).max() / 32768
    alias = np.sqrt(np.mean((stopband[inner].astype(np.float64) / 32768) ** 2))
    assert abs(peak - 0.5) < 0.005 and alias < 1e-3


def test_resampler_rejects_bad_input() -> None:
    with pytest.raises(ResampleError, match="unsupported device sample rate"):
        StreamingResampler(12345)
    resampler = StreamingResampler(48000)
    with pytest.raises(ResampleError, match="float32"):
        wrong_dtype: Any = np.zeros(4, dtype=np.float64)
        resampler.push(wrong_dtype)
    with pytest.raises(ResampleError, match="non-finite"):
        resampler.push(np.array([0.0, np.inf], dtype=np.float32))
    resampler.flush()
    with pytest.raises(ResampleError, match="already flushed"):
        resampler.push(np.zeros(4, dtype=np.float32))
    resampler.reset()
    assert resampler.push(np.zeros(4800, dtype=np.float32)).size > 0


# --- audio contract --------------------------------------------------------------------


def test_decode_chunk_accepts_f32le_and_rejects_malformed_audio() -> None:
    samples = np.array([0.0, -0.5, 0.25], dtype=np.float32)
    assert np.array_equal(decode_chunk(f32le(samples), sample_rate_hz=48000), samples)
    cases = [
        (b"", "empty"),
        (b"\x00\x00\x00", "whole number"),
        (f32le(np.array([0.0, np.nan])), "non-finite"),
        (f32le(np.zeros(2 * 48000 + 1)), "limit"),
    ]
    for body, match in cases:
        with pytest.raises(AudioContractError, match=match):
            decode_chunk(body, sample_rate_hz=48000)


# --- sessions with the toy ONNX model -------------------------------------------------------


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> LoadedOnnxModel:
    loaded = toy_checkpoint()
    metadata = model_metadata(
        loaded, source={"commit": "2" * 40, "dirty": False}, toolchain={"torch": "t"}
    )
    path = tmp_path_factory.mktemp("live") / "model.onnx"
    write_onnx_model(
        path, export_onnx_bytes(loaded, metadata), run_id="r", created_at_utc="t", environment={}
    )
    return load_onnx_model(path)


def new_session(model: LoadedOnnxModel, rate: int = 48000) -> LiveSession:
    backend = OnnxWindowBackend(create_session(model))
    return LiveSession(
        backend, StreamingSpec.from_window(model.window), model.labels, input_rate=rate
    )


def offline(model: LoadedOnnxModel, signal: Float32, rate: int) -> Any:
    backend = OnnxWindowBackend(create_session(model))
    spec = StreamingSpec.from_window(model.window)
    return classify_offline(resample_all(signal, rate), backend, spec, model.labels)


def assert_matches_offline(
    windows: list[dict[str, Any]], reference: list[Any], labels: tuple[str, ...]
) -> None:
    assert len(windows) == len(reference)
    for got, want in zip(windows, reference, strict=True):
        assert got["index"] == want.window.index and got["start_sample"] == want.window.start_sample
        assert got["label"] == labels[want.label_index] and got["confidence"] == want.confidence
        assert got["aggregate_label"] == labels[want.aggregate_label_index]
        assert got["aggregate_probabilities"] == [float(p) for p in want.aggregate_probabilities]
        assert got["inference_ms"] > 0


def test_ordered_chunks_match_the_offline_pipeline(model: LoadedOnnxModel) -> None:
    signal = tone_bursts(48000, 2.35, seed=11)
    session = new_session(model)
    windows: list[dict[str, Any]] = []
    for seq, chunk in enumerate(chunked(signal, [4800])):
        result = session.push(seq, f32le(chunk))
        assert result["seq"] == seq and result["next_seq"] == seq + 1
        windows += result["windows"]
    final = session.stop()
    windows += final["windows"]
    reference, decision = offline(model, signal, 48000)
    assert_matches_offline(windows, reference, model.labels)
    assert final["final_decision"]["label"] == decision.label
    assert final["final_decision"]["confidence"] == decision.confidence
    assert final["seconds"] == pytest.approx(math.ceil(signal.size / 3) / 16000)


def test_out_of_order_and_malformed_chunks_leave_the_session_unchanged(
    model: LoadedOnnxModel,
) -> None:
    signal = tone_bursts(48000, 0.5)
    chunks = chunked(signal, [4800])
    session = new_session(model)
    session.push(0, f32le(chunks[0]))
    with pytest.raises(ChunkOrderError, match="chunk 1 is next"):
        session.push(0, f32le(chunks[0]))  # duplicate
    with pytest.raises(ChunkOrderError, match="chunk 1 is next"):
        session.push(2, f32le(chunks[2]))  # gap / reordering
    with pytest.raises(AudioContractError):
        session.push(1, b"\x00\x00")
    assert session.next_seq == 1 and session.resampler.samples_received == 4800
    for seq in range(1, len(chunks)):
        session.push(seq, f32le(chunks[seq]))
    reference_session = new_session(model)
    for seq, chunk in enumerate(chunks):
        reference_session.push(seq, f32le(chunk))
    assert untimed(session.stop()) == untimed(reference_session.stop())


def test_reset_equals_a_fresh_session_and_stop_is_final(model: LoadedOnnxModel) -> None:
    signal = tone_bursts(48000, 1.4, seed=5)
    fresh = new_session(model)
    expected = [fresh.push(i, f32le(c)) for i, c in enumerate(chunked(signal, [4800]))]
    expected_stop = fresh.stop()
    session = new_session(model)
    session.push(0, f32le(tone_bursts(48000, 0.7, seed=9)))
    session.stop()
    with pytest.raises(SessionStateError, match="stopped"):
        session.push(1, f32le(signal[:10]))
    with pytest.raises(SessionStateError, match="already stopped"):
        session.stop()
    session.reset()
    got = [session.push(i, f32le(c)) for i, c in enumerate(chunked(signal, [4800]))]
    assert untimed(got) == untimed(expected)
    assert untimed(session.stop()) == untimed(expected_stop)


def test_empty_stream_has_no_final_decision(model: LoadedOnnxModel) -> None:
    result = new_session(model).stop()
    assert result["windows"] == [] and result["final_decision"] is None


class Scripted:
    name = "scripted"

    def __init__(self, labels: list[int]) -> None:
        self.labels = iter(labels)

    def __call__(self, windows: Float32) -> Float32:
        logits = np.zeros((1, 3), dtype=np.float32)
        logits[0, next(self.labels)] = 5.0
        return logits


def test_stable_only_when_all_five_windows_agree_with_the_aggregate() -> None:
    spec = StreamingSpec(sample_rate_hz=16000, window_samples=16000)
    session = LiveSession(
        Scripted([0] * 5 + [1] + [0] * 5), spec, ("a", "b", "c"), input_rate=16000
    )
    result = session.push(0, f32le(np.full(16000 + 10 * 1600, 0.1, dtype=np.float32)))
    stable = [w["stable"] for w in result["windows"]]
    assert stable == [False] * 4 + [True] + [False] * 5 + [True]
    assert all(w["aggregate_windows"] == min(i + 1, 5) for i, w in enumerate(result["windows"]))
    assert result["windows"][0]["level_dbfs"] == pytest.approx(20 * math.log10(3277 / 32768))


# --- HTTP API ----------------------------------------------------------------------------------


@pytest.fixture
def server(model: LoadedOnnxModel) -> Iterator[LiveDemoServer]:
    srv = LiveDemoServer(
        model, info={"model": {"onnx_sha256": model.sha256}}, port=0, max_sessions=2
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def request(
    srv: LiveDemoServer,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=30)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def post_json(srv: LiveDemoServer, path: str, payload: Any) -> tuple[int, Any]:
    status, _, body = request(
        srv, "POST", path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    return status, json.loads(body)


def post_chunk(
    srv: LiveDemoServer, session: str, seq: Any, body: bytes, **headers: str
) -> tuple[int, Any]:
    status, _, raw = request(
        srv,
        "POST",
        f"/api/sessions/{session}/chunks?seq={seq}",
        body,
        {"Content-Type": "application/octet-stream", **headers},
    )
    return status, json.loads(raw)


def test_only_loopback_addresses_can_be_bound(model: LoadedOnnxModel) -> None:
    for host in ("0.0.0.0", "192.168.1.10", "example.com", "::"):
        with pytest.raises(LiveServerError, match="loopback"):
            check_loopback(host)
        with pytest.raises(LiveServerError, match="loopback"):
            LiveDemoServer(model, info={}, host=host, port=0)
    assert check_loopback("127.0.0.1") == "127.0.0.1" and check_loopback("::1") == "::1"


def test_ui_is_local_scoped_and_locked_down(server: LiveDemoServer) -> None:
    status, headers, body = request(server, "GET", "/")
    page = body.decode("utf-8")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert "Access-Control-Allow-Origin" not in headers
    assert "Start listening" in page and "not general speech recognition" in page
    assert "not production software" in page and "Closed-set demo" in page
    static = resources.files("robust_streaming_ml.live").joinpath("static")
    for name in ("index.html", "app.js", "capture-worklet.js", "style.css"):
        text = static.joinpath(name).read_text(encoding="utf-8")
        # no external resources or endpoints (the loopback address appears only in help text)
        assert not re.search(r"https?://(?!127\.0\.0\.1\b|localhost\b)", text), name
        path = "/" if name == "index.html" else f"/{name}"
        assert request(server, "GET", path)[0] == 200
    assert '<script src="/app.js">' in page and "<script>" not in page
    status, _, body = request(server, "GET", "/api/info")
    info = json.loads(body)
    assert status == 200 and info["labels"] == ["yes", "no", "up"]
    assert 48000 in info["audio_contract"]["supported_device_rates_hz"]
    assert "stable" in info["stable_rule"] and info["streaming"]["hop_samples"] == 1600


def test_http_stream_matches_offline_and_enforces_order(
    server: LiveDemoServer, model: LoadedOnnxModel
) -> None:
    status, created = post_json(server, "/api/sessions", {"sample_rate_hz": 48000.0})
    assert status == 201 and created["next_seq"] == 0
    session = created["session"]
    signal = tone_bursts(48000, 1.9, seed=21)
    windows: list[dict[str, Any]] = []
    for seq, chunk in enumerate(chunked(signal, [4800])):
        status, result = post_chunk(server, session, seq, f32le(chunk))
        assert status == 200, result
        windows += result["windows"]
        if seq == 3:
            assert post_chunk(server, session, 3, f32le(chunk))[0] == 409  # duplicate
            assert post_chunk(server, session, 9, f32le(chunk))[0] == 409  # out of order
    status, final = post_json(server, f"/api/sessions/{session}/stop", {})
    status, _, raw = request(server, "POST", f"/api/sessions/{session}/stop")
    assert status == 409  # stop is final until reset
    windows_all = windows + final["windows"]
    reference, decision = offline(model, signal, 48000)
    assert_matches_offline(windows_all, reference, model.labels)
    assert final["final_decision"]["label"] == decision.label
    status, _, raw = request(server, "POST", f"/api/sessions/{session}/reset")
    assert status == 200 and json.loads(raw)["next_seq"] == 0
    assert post_chunk(server, session, 0, f32le(signal[:4800]))[0] == 200


@pytest.mark.parametrize(
    ("body", "headers", "seq", "expected"),
    [
        (b"\x00\x00\x00", {}, 0, 400),  # not whole float32 samples
        (f32le(np.array([np.nan, 0.0])), {}, 0, 400),  # non-finite
        (b"", {}, 0, 400),  # empty
        (f32le(np.zeros(4)), {"Content-Type": "audio/wav"}, 0, 415),
        (f32le(np.zeros(4)), {}, "x", 400),  # bad sequence number
        (f32le(np.zeros(2 * 48000 + 1)), {}, 0, 413),  # over 2 s
    ],
    ids=["partial-sample", "non-finite", "empty", "wrong-type", "bad-seq", "too-long"],
)
def test_malformed_audio_is_rejected(
    server: LiveDemoServer, body: bytes, headers: dict[str, str], seq: Any, expected: int
) -> None:
    _, created = post_json(server, "/api/sessions", {"sample_rate_hz": 48000})
    status, error = post_chunk(server, created["session"], seq, body, **headers)
    assert status == expected and "error" in error
    assert post_chunk(server, created["session"], 0, f32le(np.zeros(480)))[0] == 200


def test_api_errors_and_cross_site_protection(server: LiveDemoServer) -> None:
    assert post_json(server, "/api/sessions", {"sample_rate_hz": 12345})[0] == 400
    assert post_json(server, "/api/sessions", {"rate": 48000})[0] == 400
    assert post_json(server, "/api/sessions", {"sample_rate_hz": "48000"})[0] == 400
    assert post_chunk(server, "nope", 0, f32le(np.zeros(4)))[0] == 404
    assert request(server, "GET", "/api/sessions")[0] == 405
    assert request(server, "GET", "/nowhere")[0] == 404
    evil_host = request(server, "GET", "/", headers={"Host": "attacker.example:80"})
    assert evil_host[0] == 403
    status, created = post_json(server, "/api/sessions", {"sample_rate_hz": 48000})
    session = created["session"]
    status, _ = post_chunk(
        server, session, 0, f32le(np.zeros(8)), Origin="https://attacker.example"
    )
    assert status == 403
    same_origin = f"http://127.0.0.1:{server.port}"
    assert post_chunk(server, session, 0, f32le(np.zeros(8)), Origin=same_origin)[0] == 200
    assert request(server, "DELETE", f"/api/sessions/{session}")[0] == 200
    assert request(server, "DELETE", f"/api/sessions/{session}")[0] == 404


def test_least_recently_used_sessions_are_evicted(server: LiveDemoServer) -> None:
    ids = [
        post_json(server, "/api/sessions", {"sample_rate_hz": 16000})[1]["session"]
        for _ in range(3)
    ]
    assert post_chunk(server, ids[0], 0, f32le(np.zeros(8)))[0] == 404  # max_sessions=2
    assert post_chunk(server, ids[2], 0, f32le(np.zeros(8)))[0] == 200


def test_nothing_is_written_to_disk(
    server: LiveDemoServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _, created = post_json(server, "/api/sessions", {"sample_rate_hz": 48000})
    post_chunk(server, created["session"], 0, f32le(tone_bursts(48000, 1.2)))
    post_json(server, f"/api/sessions/{created['session']}/stop", {})
    assert list(tmp_path.iterdir()) == []
