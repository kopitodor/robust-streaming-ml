from __future__ import annotations

import numpy as np
import pytest
import torch

from robust_streaming_ml.data.manifest import ManifestRow
from robust_streaming_ml.evaluation.metrics import softmax
from robust_streaming_ml.evaluation.predictions import (
    PredictionError,
    PredictionSet,
    decode_predictions,
    encode_calibrated,
    encode_predictions,
    predict_split,
)
from robust_streaming_ml.features import LogMelSpectrogram
from robust_streaming_ml.model import KeywordCNN
from robust_streaming_ml.training.config import FeatureSection, ModelSection, WindowSection
from robust_streaming_ml.training.data import SplitAudio

LABELS = ("yes", "no", "up")


def prediction_set(split: str = "validation") -> PredictionSet:
    logits = np.array(
        [[0.1, 2.5, -1.0], [3.25, 3.25, 0.0], [-0.3333, 0.1, 7.0], [1e-7, -2.0, 0.5]],
        dtype=np.float32,
    )
    return PredictionSet(
        split=split,
        labels=LABELS,
        relative_paths=("no/aaaaaaaa_nohash_0.wav", "no/bbbbbbbb_nohash_0.wav",
                        "up/aaaaaaaa_nohash_0.wav", "yes/cccccccc_nohash_1.wav"),
        audio_sha256=("1" * 64, "2" * 64, "3" * 64, "1" * 64),
        speaker_ids=("aaaaaaaa", "bbbbbbbb", "aaaaaaaa", "cccccccc"),
        y_true=np.array([1, 0, 2, 0], dtype=np.int64),
        logits=logits,
    )  # fmt: skip


def test_round_trip_is_exact_and_ordered() -> None:
    original = prediction_set()
    decoded = decode_predictions(encode_predictions(original), split="validation", labels=LABELS)
    assert decoded.relative_paths == original.relative_paths
    assert decoded.audio_sha256 == original.audio_sha256
    assert decoded.speaker_ids == original.speaker_ids
    assert np.array_equal(decoded.y_true, original.y_true)
    assert decoded.logits.dtype == np.float32 and np.array_equal(decoded.logits, original.logits)
    assert decoded.y_pred.tolist() == [1, 0, 2, 2]  # row 2 tie -> lowest index


def test_encoded_columns() -> None:
    text = encode_predictions(prediction_set()).decode("utf-8")
    header, first = text.splitlines()[:2]
    assert header.split(",")[:8] == [
        "relative_path", "audio_sha256", "speaker_id", "label", "label_index",
        "pred_index", "pred_label", "correct",
    ]  # fmt: skip
    assert header.endswith("prob_up")
    fields = first.split(",")
    assert fields[:8] == [
        "no/aaaaaaaa_nohash_0.wav",
        "1" * 64,
        "aaaaaaaa",
        "no",
        "1",
        "1",
        "no",
        "true",
    ]
    assert float(fields[8]) == float(np.float32(0.1))


def test_calibrated_file_uses_the_temperature() -> None:
    preds = prediction_set()
    lines = encode_calibrated(preds, 2.0).decode("utf-8").splitlines()
    assert lines[0] == "relative_path,pred_index,cal_prob_yes,cal_prob_no,cal_prob_up"
    values = [float(v) for v in lines[1].split(",")[2:]]
    np.testing.assert_array_equal(values, softmax(preds.logits, 2.0)[0])


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (",1,1,no,true,", ",1,0,yes,false,", "argmax"),
        (",1,1,no,true,", ",1,1,no,false,", "correct column"),
        (",1,1,no,true,", ",1,1,up,true,", "label columns"),
        ("relative_path", "path", "header"),
    ],
)
def test_tampered_artifacts_are_rejected(old: str, new: str, message: str) -> None:
    text = encode_predictions(prediction_set()).decode("utf-8").replace(old, new, 1)
    with pytest.raises(PredictionError, match=message):
        decode_predictions(text.encode("utf-8"), split="validation", labels=LABELS)


def test_tampered_probability_or_non_float32_logit_is_rejected() -> None:
    lines = encode_predictions(prediction_set()).decode("utf-8").splitlines()
    fields = lines[1].split(",")
    fields[-1] = repr(float(fields[-1]) + 1e-12)
    bad_prob = "\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n"
    with pytest.raises(PredictionError, match="probabilities"):
        decode_predictions(bad_prob.encode("utf-8"), split="validation", labels=LABELS)
    fields = lines[1].split(",")
    fields[8] = "0.1"  # not exactly representable as float32
    bad_logit = "\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n"
    with pytest.raises(PredictionError, match="float32"):
        decode_predictions(bad_logit.encode("utf-8"), split="validation", labels=LABELS)


def test_prediction_set_validation_and_subset() -> None:
    preds = prediction_set()
    sub = preds.subset([0, 2])
    assert sub.relative_paths == (preds.relative_paths[0], preds.relative_paths[2])
    assert np.array_equal(sub.logits, preds.logits[[0, 2]])
    with pytest.raises(PredictionError):
        PredictionSet(**{**preds.__dict__, "split": "dev"})
    with pytest.raises(PredictionError):
        PredictionSet(**{**preds.__dict__, "logits": preds.logits.astype(np.float64)})


def _rows(paths: tuple[str, ...], split: str) -> list[ManifestRow]:
    return [
        ManifestRow(
            relative_path=p, split=split, label=p.split("/")[0], label_index=0,
            speaker_id="aaaaaaaa", utterance_index=i, sha256="a" * 64, audio_sha256=f"{i:064x}",
            size_bytes=32044, sample_rate_hz=16000, num_channels=1, sample_format="PCM_16",
            num_frames=16000, duration_s="1.000000", peak_abs=1, is_all_zero=False,
        )
        for i, p in enumerate(paths)
    ]  # fmt: skip


def test_inference_is_deterministic_ordered_and_batch_independent() -> None:
    torch.manual_seed(0)
    features = FeatureSection(n_mels=16)
    model = KeywordCNN(ModelSection(channels=[4, 8]), num_classes=3, n_mels=16)
    frontend = LogMelSpectrogram(features, WindowSection())
    gen = torch.Generator().manual_seed(1)
    waves = (torch.randn(5, 16000, generator=gen) * 3000).to(torch.int16)
    paths = tuple(f"yes/{i:08x}_nohash_0.wav" for i in range(5))
    audio = SplitAudio("test", waves, torch.tensor([0, 1, 2, 0, 1]), paths)
    rows = _rows(paths, "test")
    a = predict_split(model, frontend, audio, rows, LABELS, batch_size=2)
    b = predict_split(model, frontend, audio, rows, LABELS, batch_size=2)
    c = predict_split(model, frontend, audio, rows, LABELS, batch_size=5)
    assert a.relative_paths == paths and a.split == "test"
    assert np.array_equal(a.logits, b.logits)
    np.testing.assert_allclose(a.logits, c.logits, rtol=1e-5, atol=1e-5)
    assert not model.training
    with pytest.raises(PredictionError, match="same order"):
        predict_split(model, frontend, audio, list(reversed(rows)), LABELS, batch_size=2)
    with pytest.raises(PredictionError, match="another split"):
        predict_split(model, frontend, audio, _rows(paths, "validation"), LABELS, batch_size=2)
