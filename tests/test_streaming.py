"""Streaming contract kw10-streaming-v1: windows, chunking, padding, reset, and causality."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from robust_streaming_ml.evaluation.predictions import infer_logits
from robust_streaming_ml.features import LogMelSpectrogram, apply_window, pcm16_to_float
from robust_streaming_ml.model import KeywordCNN
from robust_streaming_ml.streaming.classifier import (
    StreamingClassifier,
    TorchWindowBackend,
    WindowPrediction,
    WindowPredictor,
    classify_offline,
    classify_stream,
)
from robust_streaming_ml.streaming.windowing import (
    AGGREGATION_WINDOWS,
    HOP_SAMPLES,
    STREAMING_PROTOCOL_ID,
    StreamingError,
    StreamingSpec,
    StreamingWindower,
    Window,
    offline_windows,
    stream_samples,
)
from robust_streaming_ml.training.config import FeatureSection, ModelSection, WindowSection

SMALL = StreamingSpec(sample_rate_hz=16000, window_samples=10, hop_samples=4, aggregation_windows=3)
LABELS = ("a", "b", "c")
Float32 = npt.NDArray[np.float32]


def ramp(n: int, start: int = 1) -> npt.NDArray[np.int16]:
    """Distinct non-zero samples, so every window's content identifies its position."""
    return ((np.arange(n, dtype=np.int64) + start) % 30000 + 1).astype(np.int16)


class ContentBackend:
    """Deterministic logits computed from the window content; records batch sizes."""

    name = "content"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, windows: Float32) -> Float32:
        self.batch_sizes.append(int(windows.shape[0]))
        w = windows.astype(np.float64) * 100.0
        half = windows.shape[1] // 2
        logits = np.stack([w.mean(1), w[:, :half].mean(1), -w[:, half:].mean(1)], axis=1)
        return logits.astype(np.float32)


class ScriptedBackend:
    """Returns preset logits, one row per call."""

    name = "scripted"

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows: Iterator[list[float]] = iter(rows)

    def __call__(self, windows: Float32) -> Float32:
        return np.asarray([next(self.rows)], dtype=np.float32)


def random_chunks(
    signal: npt.NDArray[Any], rng: np.random.Generator, max_size: int
) -> list[npt.NDArray[Any]]:
    chunks, pos = [], 0
    while pos < signal.size:
        size = int(rng.integers(0, max_size + 1))  # includes empty chunks
        chunks.append(signal[pos : pos + size])
        pos += size
    return chunks


def windower_windows(chunks: list[npt.NDArray[Any]], spec: StreamingSpec = SMALL) -> list[Window]:
    windower = StreamingWindower(spec, sample_rate_hz=spec.sample_rate_hz)
    out: list[Window] = []
    for chunk in chunks:
        out.extend(windower.push(chunk))
    return out + windower.flush()


def same(a: Window, b: Window) -> bool:
    return (
        (a.index, a.start_sample, a.valid_samples, a.final_partial, a.emitted_at_sample)
        == (b.index, b.start_sample, b.valid_samples, b.final_partial, b.emitted_at_sample)
    ) and np.array_equal(a.samples, b.samples)


def same_prediction(a: WindowPrediction, b: WindowPrediction) -> bool:
    return (
        same(a.window, b.window)
        and np.array_equal(a.logits, b.logits)
        and np.array_equal(a.probabilities, b.probabilities)
        and np.array_equal(a.aggregate_probabilities, b.aggregate_probabilities)
        and (a.label_index, a.aggregate_label_index, a.aggregate_windows)
        == (b.label_index, b.aggregate_label_index, b.aggregate_windows)
    )


# --- protocol constants and input contract ----------------------------------------------


def test_protocol_constants_bind_to_the_checkpoint_window() -> None:
    spec = StreamingSpec.from_window(WindowSection())
    assert (spec.sample_rate_hz, spec.window_samples) == (16000, 16000)
    assert (spec.hop_samples, spec.aggregation_windows) == (HOP_SAMPLES, AGGREGATION_WINDOWS)
    assert (HOP_SAMPLES, AGGREGATION_WINDOWS) == (1600, 5)
    described = spec.describe()
    assert described["protocol"] == STREAMING_PROTOCOL_ID == "kw10-streaming-v1"
    assert described["startup_padding"] == "none" and described["inference_batch_size"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hop_samples": 11},
        {"hop_samples": 0},
        {"aggregation_windows": 0},
        {"window_samples": 0},
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict[str, int]) -> None:
    settings: dict[str, Any] = {
        "sample_rate_hz": 16000,
        "window_samples": 10,
        "hop_samples": 4,
        **kwargs,
    }
    with pytest.raises(StreamingError):
        StreamingSpec(**settings)


def test_int16_chunks_are_scaled_exactly_like_the_front_end() -> None:
    samples = np.array([-32768, -1, 0, 1, 12345, 32767], dtype=np.int16)
    expected = pcm16_to_float(torch.from_numpy(samples)).numpy()
    got = stream_samples(samples)
    assert got.dtype == np.float32 and np.array_equal(got, expected)


def test_float32_chunks_are_used_as_they_are() -> None:
    samples = np.array([0.5, -1.25, 3.0], dtype=np.float32)  # above full scale is kept
    got = stream_samples(samples)
    assert np.array_equal(got, samples) and got is not samples


@pytest.mark.parametrize(
    "chunk",
    [
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.int32),
        np.zeros((2, 3), dtype=np.int16),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        [1, 2, 3],
    ],
)
def test_other_inputs_are_rejected_not_converted(chunk: Any) -> None:
    with pytest.raises(StreamingError):
        stream_samples(chunk)


def test_declared_sample_rate_must_match_the_model() -> None:
    with pytest.raises(StreamingError, match="resampling"):
        StreamingWindower(SMALL, sample_rate_hz=8000)
    with pytest.raises(StreamingError, match="resampling"):
        StreamingClassifier(ContentBackend(), SMALL, LABELS, sample_rate_hz=44100)


# --- window boundaries, emission, padding ------------------------------------------------


def test_no_startup_padding_and_emission_exactly_when_complete() -> None:
    windower = StreamingWindower(SMALL, sample_rate_hz=16000)
    signal = ramp(30)
    appeared: dict[int, int] = {}
    for i in range(signal.size):
        for window in windower.push(signal[i : i + 1]):
            appeared[window.index] = i + 1
    # window k needs samples [4k, 4k + 10): complete after sample 4k + 10
    assert appeared == {k: 4 * k + 10 for k in range(6)}
    final = windower.flush()
    assert [w.index for w in final] == []  # 30 = 5 * 4 + 10: fully covered


def test_windows_cover_the_grid_with_exact_timestamps() -> None:
    windows = windower_windows([ramp(23)])
    assert [(w.index, w.start_sample, w.end_sample) for w in windows] == [
        (0, 0, 10),
        (1, 4, 14),
        (2, 8, 18),
        (3, 12, 22),
        (4, 16, 26),
    ]
    assert [w.valid_samples for w in windows] == [10, 10, 10, 10, 7]
    assert [w.final_partial for w in windows] == [False] * 4 + [True]
    assert [w.emitted_at_sample for w in windows] == [10, 14, 18, 22, 23]
    timing = windows[1].timing(16000)
    assert timing["start_s"] == 4 / 16000 and timing["end_s"] == 14 / 16000


@pytest.mark.parametrize(
    ("n", "starts", "final_valid"),
    [
        (10, [0], None),  # exactly one window, nothing uncovered
        (11, [0, 4], 7),  # one uncovered sample -> one final window
        (13, [0, 4], 9),
        (14, [0, 4], None),  # second window complete
        (15, [0, 4, 8], 7),
    ],
)
def test_exactly_one_final_partial_window_when_samples_are_uncovered(
    n: int, starts: list[int], final_valid: int | None
) -> None:
    signal = ramp(n)
    windows = windower_windows([signal])
    assert [w.start_sample for w in windows] == starts
    finals = [w for w in windows if w.final_partial]
    if final_valid is None:
        assert finals == []
        return
    (final,) = finals
    assert final is windows[-1] and final.valid_samples == final_valid == n - final.start_sample
    expected = np.zeros(10, dtype=np.float32)
    expected[:final_valid] = stream_samples(signal[final.start_sample :])
    assert np.array_equal(final.samples, expected)


def test_every_sample_is_covered_and_no_redundant_window_is_emitted() -> None:
    for n in range(0, 60):
        windows = offline_windows(ramp(n), SMALL)
        covered = np.zeros(n, dtype=bool)
        for w in windows:
            covered[w.start_sample : w.start_sample + w.valid_samples] = True
        assert covered.all()
        if windows and windows[-1].final_partial:
            before = windows[-2].end_sample if len(windows) > 1 else 0
            assert n > before  # the final window adds uncovered samples


def test_short_audio_is_one_zero_padded_clip_level_window() -> None:
    spec = StreamingSpec.from_window(WindowSection())
    clip = ramp(1234)
    windows = windower_windows([clip[:500], clip[500:]], spec)
    (window,) = windows
    clip_level = pcm16_to_float(torch.from_numpy(apply_window(clip, WindowSection()))).numpy()
    assert window.final_partial and window.valid_samples == 1234 and window.start_sample == 0
    assert np.array_equal(window.samples, clip_level)


def test_a_full_length_clip_is_one_full_window() -> None:
    spec = StreamingSpec.from_window(WindowSection())
    (window,) = windower_windows([ramp(16000)], spec)
    assert not window.final_partial and window.valid_samples == 16000


def test_empty_stream_has_no_window_and_no_decision() -> None:
    assert windower_windows([np.zeros(0, dtype=np.int16)]) == []
    classifier = StreamingClassifier(ContentBackend(), SMALL, LABELS, sample_rate_hz=16000)
    assert classifier.push(np.zeros(0, dtype=np.int16)) == []
    assert classifier.flush() == []
    assert classifier.decision() is None


# --- chunking invariance -------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 9, 10, 11, 13, 14, 37, 100])
def test_any_chunking_gives_the_offline_windows(n: int) -> None:
    signal = ramp(n)
    reference = offline_windows(signal, SMALL)
    rng = np.random.default_rng(n)
    chunkings = [[signal], [signal[i : i + 1] for i in range(n)]]
    chunkings += [random_chunks(signal, rng, size) for size in (2, 5, 13, 40) for _ in range(5)]
    for chunks in chunkings:
        got = windower_windows(chunks)
        assert len(got) == len(reference) and all(
            same(a, b) for a, b in zip(got, reference, strict=True)
        )


def test_int16_and_equivalent_float32_streams_give_identical_windows() -> None:
    signal = ramp(47)
    as_float = stream_samples(signal)
    a = windower_windows([signal[:20], signal[20:]])
    b = windower_windows([as_float[:7], as_float[7:]])
    assert all(same(x, y) for x, y in zip(a, b, strict=True))


def test_windows_are_read_only_copies() -> None:
    signal = ramp(12)
    (first, second) = windower_windows([signal])
    with pytest.raises(ValueError, match="read-only"):
        first.samples[0] = 0
    assert not np.shares_memory(first.samples, second.samples)


def test_buffer_holds_at_most_one_window_plus_one_chunk() -> None:
    windower = StreamingWindower(SMALL, sample_rate_hz=16000)
    signal = ramp(500)
    for start in range(0, 500, 7):
        windower.push(signal[start : start + 7])
        assert windower.buffered_samples < SMALL.window_samples + 7
    assert windower.samples_received == 500 and windower.windows_emitted == 123


# --- call order and reset ------------------------------------------------------------------


def test_flush_ends_the_stream() -> None:
    windower = StreamingWindower(SMALL, sample_rate_hz=16000)
    windower.push(ramp(5))
    windower.flush()
    assert windower.finished
    with pytest.raises(StreamingError, match="already flushed"):
        windower.push(ramp(1))
    with pytest.raises(StreamingError, match="already flushed"):
        windower.flush()
    classifier = StreamingClassifier(ContentBackend(), SMALL, LABELS, sample_rate_hz=16000)
    classifier.push(ramp(12))
    with pytest.raises(StreamingError, match="after flush"):
        classifier.decision()


def test_reset_gives_the_same_outputs_as_a_new_instance() -> None:
    signal = ramp(37, start=900)
    fresh = StreamingClassifier(ContentBackend(), SMALL, LABELS, sample_rate_hz=16000)
    expected = fresh.push(signal) + fresh.flush()
    reused = StreamingClassifier(ContentBackend(), SMALL, LABELS, sample_rate_hz=16000)
    reused.push(ramp(23, start=5))  # mid-stream: buffered samples and aggregation history
    reused.reset()
    got = reused.push(signal[:11]) + reused.push(signal[11:]) + reused.flush()
    assert len(got) == len(expected) and all(
        same_prediction(a, b) for a, b in zip(got, expected, strict=True)
    )
    assert got[0].window.start_sample == 0  # timestamps restart at 0
    reused.reset()  # reset after flush also starts a new stream
    assert reused.push(signal) and reused.windower.samples_received == 37


# --- causality -----------------------------------------------------------------------------


def test_outputs_do_not_depend_on_future_audio() -> None:
    signal = ramp(60)
    altered = signal.copy()
    t = 30
    altered[t:] = -altered[t:]
    base = classify_offline(signal, ContentBackend(), SMALL, LABELS)[0]
    other = classify_offline(altered, ContentBackend(), SMALL, LABELS)[0]
    past = [p for p in base if p.window.end_sample <= t]
    assert len(past) == 6
    assert all(same_prediction(a, b) for a, b in zip(past, other, strict=False))
    # the first window that sees the change differs
    assert not np.array_equal(base[len(past)].logits, other[len(past)].logits)


def test_emitted_outputs_never_change_afterwards() -> None:
    classifier = StreamingClassifier(ContentBackend(), SMALL, LABELS, sample_rate_hz=16000)
    first = classifier.push(ramp(20))
    snapshot = [(p.logits.copy(), p.aggregate_probabilities.copy()) for p in first]
    classifier.push(ramp(40, start=300))
    classifier.flush()
    for p, (logits, aggregate) in zip(first, snapshot, strict=True):
        assert np.array_equal(p.logits, logits)
        assert np.array_equal(p.aggregate_probabilities, aggregate)


def test_aggregate_is_the_causal_mean_of_the_last_m_windows() -> None:
    rows = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    predictor = WindowPredictor(ScriptedBackend(rows), SMALL, LABELS)
    windows = offline_windows(ramp(26), SMALL)
    assert len(windows) == 5
    predictions = [predictor.predict(w) for w in windows]
    probs = [np.exp(r) / np.exp(r).sum() for r in np.asarray(rows, dtype=np.float64)]
    for k, p in enumerate(predictions):
        recent = probs[max(0, k - 2) : k + 1]
        assert p.aggregate_windows == len(recent)
        np.testing.assert_allclose(p.aggregate_probabilities, np.mean(recent, axis=0), rtol=1e-12)
        assert p.aggregate_label_index == int(np.argmax(np.mean(recent, axis=0)))
    assert [p.label_index for p in predictions] == [0, 1, 2, 0, 0]  # ties -> lowest index
    decision = predictor.decision()
    assert decision is not None and decision.windows == 5 and decision.last_window_index == 4
    assert decision.label_index == predictions[-1].aggregate_label_index


def test_every_window_is_inferred_alone() -> None:
    backend = ContentBackend()
    classifier = StreamingClassifier(backend, SMALL, LABELS, sample_rate_hz=16000)
    emitted = classifier.push(ramp(40)) + classifier.flush()  # one push emits 8 windows
    assert len(emitted) == 9 and backend.batch_sizes == [1] * 9


@pytest.mark.parametrize(
    "logits",
    [
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.array([[0.0, np.nan, 0.0]], dtype=np.float32),
    ],
)
def test_invalid_backend_outputs_fail(logits: npt.NDArray[Any]) -> None:
    class Bad:
        name = "bad"

        def __call__(self, windows: Float32) -> Any:
            return logits

    classifier = StreamingClassifier(Bad(), SMALL, LABELS, sample_rate_hz=16000)
    with pytest.raises(StreamingError, match="backend bad"):
        classifier.push(ramp(10))


def test_prediction_description_names_labels_and_times() -> None:
    (prediction,) = classify_offline(ramp(10), ContentBackend(), SMALL, LABELS)[0]
    described = prediction.describe(LABELS, 16000)
    assert described["label"] in LABELS and described["inference_ms"] is None
    assert described["end_sample"] == 10


# --- with the real front end and a small CNN ---------------------------------------------


@pytest.fixture(scope="module")
def torch_backend() -> TorchWindowBackend:
    torch.manual_seed(3)
    features = FeatureSection(n_mels=16)
    model = KeywordCNN(ModelSection(channels=[4, 8]), num_classes=3, n_mels=16).eval()
    return TorchWindowBackend(model, LogMelSpectrogram(features, WindowSection()))


def test_torch_streaming_is_bit_identical_to_offline(torch_backend: TorchWindowBackend) -> None:
    spec = StreamingSpec.from_window(WindowSection())
    signal = (np.random.default_rng(7).standard_normal(37123) * 4000).astype(np.int16)
    reference, decision = classify_offline(signal, torch_backend, spec, LABELS)
    assert len(reference) == 15 and reference[-1].window.final_partial
    rng = np.random.default_rng(11)
    for max_size in (1, 999, 20000, 40000):
        chunks = random_chunks(signal, rng, max_size) if max_size > 1 else [signal]
        got, got_decision = classify_stream(
            chunks, torch_backend, spec, LABELS, sample_rate_hz=16000
        )
        assert all(same_prediction(a, b) for a, b in zip(got, reference, strict=True))
        assert got_decision is not None and decision is not None
        assert (got_decision.label_index, got_decision.windows) == (decision.label_index, 15)
        assert np.array_equal(got_decision.probabilities, decision.probabilities)


def test_one_window_stream_decision_is_the_clip_level_prediction(
    torch_backend: TorchWindowBackend,
) -> None:
    spec = StreamingSpec.from_window(WindowSection())
    clip = (np.random.default_rng(5).standard_normal(9000) * 3000).astype(np.int16)
    predictions, decision = classify_stream(
        [clip[:4000], clip[4000:]], torch_backend, spec, LABELS, sample_rate_hz=16000
    )
    window = torch.from_numpy(apply_window(clip, WindowSection())).unsqueeze(0)
    clip_logits = infer_logits(
        torch_backend.model,
        torch_backend.frontend,
        window,
        batch_size=1,
        num_classes=3,
        to_float=pcm16_to_float,
    )
    assert len(predictions) == 1 and np.array_equal(predictions[0].logits, clip_logits[0])
    assert decision is not None and decision.label_index == int(np.argmax(clip_logits[0]))
    assert decision.label == LABELS[decision.label_index]
