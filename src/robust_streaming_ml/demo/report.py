"""Self-contained HTML page for one demo run (no scripts, no external resources).

Every value from the record is HTML-escaped. Audio, when embedded, is a
``data:`` URI of the local file (clean) or of a float32 WAV of the corrupted
samples, so the page works offline and sends nothing anywhere.
"""

from __future__ import annotations

import base64
import html
import json
from typing import Any

import numpy as np
import numpy.typing as npt

from robust_streaming_ml.data.wav import encode_wav_float32

__all__ = ["LABEL_COLORS", "render_html"]

LABEL_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#8c8c8c",
)

_CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --card: #f6f8fa; --line: #d1d9e0;
  --warn-bg: #fff8c5; --warn-fg: #3b2300; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --card: #161b22; --line: #30363d;
    --warn-bg: #3b2e00; --warn-fg: #f5e0a8; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 8px; }
h2 { font-size: 1.15rem; margin: 32px 0 8px; border-bottom: 1px solid var(--line);
  padding-bottom: 4px; }
.warn { background: var(--warn-bg); color: var(--warn-fg); padding: 10px 12px;
  border-radius: 6px; }
.muted { color: var(--muted); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 14px; }
.card .label { font-size: 1.8rem; font-weight: 650; }
.swatch { display: inline-block; width: 0.8em; height: 0.8em; border-radius: 2px;
  margin-right: 4px; vertical-align: -0.05em; }
.legend span.item { margin-right: 12px; white-space: nowrap; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 6px; text-align: left; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
code { font-size: 12.5px; }
svg text { fill: var(--muted); font-size: 11px; }
audio { width: 100%; max-width: 480px; }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 10px; overflow-x: auto; font-size: 12px; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _color(labels: list[str], label: str) -> str:
    return LABEL_COLORS[labels.index(label) % len(LABEL_COLORS)]


def _legend(labels: list[str]) -> str:
    items = "".join(
        f'<span class="item"><span class="swatch" style="background:{_color(labels, lab)}">'
        f"</span>{_e(lab)}</span>"
        for lab in labels
    )
    return f'<p class="legend">{items}</p>'


def _timeline(
    run: dict[str, Any], labels: list[str], duration_s: float, hop_s: float, title: str
) -> str:
    windows = run["windows"]
    if not windows:
        return f"<p><strong>{_e(title)}:</strong> no window (empty stream).</p>"
    width, height, top, bar_h, strip_h = 1000.0, 190.0, 10.0, 120.0, 14.0
    span = max(duration_s, windows[-1]["end_s"])
    scale = (width - 60.0) / span
    bar_w = max(1.0, 0.8 * hop_s * scale)
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" role="img" '
        f'aria-label="{_e(title)}">',
        f'<line x1="40" y1="{top + bar_h}" x2="{width - 10}" y2="{top + bar_h}" '
        'stroke="currentColor" stroke-opacity="0.3"/>',
        f'<text x="0" y="{top + 10}">1.0</text><text x="0" y="{top + bar_h}">0.0</text>',
    ]
    points = []
    for w in windows:
        x = 40.0 + w["start_s"] * scale
        h = w["confidence"] * bar_h
        tip = (
            f"window {w['index']}, {w['start_s']:.2f}-{w['end_s']:.2f} s: {w['label']} "
            f"({w['confidence']:.3f}); aggregate {w['aggregate_label']} "
            f"({w['aggregate_confidence']:.3f})"
            + (" [final, zero-padded]" if w["final_partial"] else "")
        )
        parts.append(
            f'<rect x="{x:.2f}" y="{top + bar_h - h:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
            f'fill="{_color(labels, w["label"])}"><title>{_e(tip)}</title></rect>'
        )
        parts.append(
            f'<rect x="{x:.2f}" y="{top + bar_h + 6:.2f}" width="{bar_w:.2f}" '
            f'height="{strip_h}" fill="{_color(labels, w["aggregate_label"])}">'
            f"<title>{_e(tip)}</title></rect>"
        )
        points.append(f"{x + bar_w / 2:.2f},{top + bar_h - w['aggregate_confidence'] * bar_h:.2f}")
    parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" stroke="currentColor" '
        'stroke-width="1.5" stroke-opacity="0.7"/>'
    )
    for second in range(int(span) + 1):
        x = 40.0 + second * scale
        parts.append(f'<text x="{x:.1f}" y="{height - 8}">{second}s</text>')
    parts.append(
        f'<text x="{width - 10}" y="{top + bar_h + 18}" text-anchor="end">aggregate</text></svg>'
    )
    return (
        f"<h3>{_e(title)}</h3>"
        '<p class="muted">Bars: each window\'s top-1 confidence, coloured by its label, at the '
        "window's start time. Line: the causal aggregate's confidence. Strip: the aggregate "
        "label. Hover for details.</p>" + "".join(parts)
    )


def _decision_card(title: str, run: dict[str, Any] | None, labels: list[str]) -> str:
    if run is None:
        return ""
    decision = run["decision"]
    if decision is None:
        body = '<div class="label">no decision</div><div class="muted">empty stream</div>'
    else:
        body = (
            f'<div class="label"><span class="swatch" '
            f'style="background:{_color(labels, decision["label"])}"></span>'
            f"{_e(decision['label'])}</div>"
            f"<div>confidence {decision['confidence']:.3f} (aggregate of the last "
            f"{min(decision['windows'], 5)} of {decision['windows']} window(s))</div>"
        )
    return f'<div class="card"><div class="muted">{_e(title)}</div>{body}</div>'


def _window_table(record: dict[str, Any]) -> str:
    clean = record["runs"]["clean"]["windows"]
    corrupted_run = record["runs"]["corrupted"]
    corrupted = corrupted_run["windows"] if corrupted_run else None
    head = (
        "<tr><th class=num>#</th><th class=num>start s</th><th class=num>end s</th>"
        "<th class=num>valid</th><th>window label</th><th class=num>conf</th>"
        "<th>aggregate</th><th class=num>agg conf</th><th class=num>ms</th>"
    )
    if corrupted is not None:
        head += (
            "<th>corrupted label</th><th class=num>conf</th><th>corrupted aggregate</th>"
            "<th class=num>agg conf</th>"
        )
    rows = []
    for i, w in enumerate(clean):
        cells = (
            f"<td class=num>{w['index']}</td><td class=num>{w['start_s']:.2f}</td>"
            f"<td class=num>{w['end_s']:.2f}</td><td class=num>{w['valid_samples']}"
            f"{' (padded)' if w['final_partial'] else ''}</td><td>{_e(w['label'])}</td>"
            f"<td class=num>{w['confidence']:.3f}</td><td>{_e(w['aggregate_label'])}</td>"
            f"<td class=num>{w['aggregate_confidence']:.3f}</td>"
            f"<td class=num>{w['inference_ms']:.2f}</td>"
        )
        if corrupted is not None:
            c = corrupted[i]
            cells += (
                f"<td>{_e(c['label'])}</td><td class=num>{c['confidence']:.3f}</td>"
                f"<td>{_e(c['aggregate_label'])}</td>"
                f"<td class=num>{c['aggregate_confidence']:.3f}</td>"
            )
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="scroll"><table>{head}</tr>{"".join(rows)}</table></div>'


def _dl(items: list[tuple[str, Any]]) -> str:
    return "<dl>" + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in items) + "</dl>"


def _audio(title: str, data: bytes) -> str:
    uri = "data:audio/wav;base64," + base64.b64encode(data).decode("ascii")
    return f'<p>{_e(title)}<br><audio controls preload="none" src="{uri}"></audio></p>'


def render_html(
    record: dict[str, Any],
    *,
    clean_wav: bytes | None,
    corrupted_samples: npt.NDArray[np.float32] | None,
    sample_rate_hz: int,
) -> str:
    labels = list(record["model"]["labels"])
    model = record["model"]
    runs = record["runs"]
    duration = float(record["input"]["duration_s"])
    corruption = record["corruption"]
    source = record["source"]
    title_corruption = f"{corruption['name']} / {corruption['severity']}" if corruption else "none"
    sections = [
        "<header><h1>Streaming keyword-spotting demo</h1>"
        f'<p class="warn"><strong>Local demo output for one file: not a measured result and '
        "not production-ready.</strong> Closed set: every window is assigned one of the 10 "
        "keywords, including silence, noise, and other speech; there is no silence or "
        "unknown-word rejection.</p>"
        f'<p class="muted">Input <code>{_e(record["input"]["file"])}</code>: '
        f"{duration:.2f} s, {record['input']['samples']} samples at {sample_rate_hz} Hz. "
        f"Corruption: {_e(title_corruption)}. Created {_e(record['created_at_utc'])}.</p>"
        "</header>",
        '<h2>Stream decision</h2><div class="cards">'
        + _decision_card("clean input", runs["clean"], labels)
        + _decision_card(f"corrupted: {title_corruption}", runs["corrupted"], labels)
        + "</div>",
    ]
    comparison = record.get("comparison")
    if comparison:
        sections.append(
            f"<p>Window labels unchanged by the corruption: {comparison['same_window_label']} "
            f"of {comparison['windows']}. Same stream decision: "
            f"{'yes' if comparison['same_decision'] else 'no'}.</p>"
        )
    if clean_wav is not None or corrupted_samples is not None:
        audio = "<h2>Listen</h2>"
        if clean_wav is not None:
            audio += _audio("Clean input", clean_wav)
        if corrupted_samples is not None:
            audio += _audio(
                "Corrupted (float32; values beyond full scale are clipped by the player)",
                encode_wav_float32(corrupted_samples, sample_rate_hz),
            )
        sections.append(audio)
    sections.append("<h2>Timeline</h2>" + _legend(labels))
    hop_s = record["streaming"]["hop_samples"] / sample_rate_hz
    sections.append(_timeline(runs["clean"], labels, duration, hop_s, "Clean input"))
    if runs["corrupted"]:
        sections.append(
            _timeline(runs["corrupted"], labels, duration, hop_s, f"Corrupted: {title_corruption}")
        )
    sections.append("<h2>Windows</h2>" + _window_table(record))
    timing = runs["clean"]["timing"]
    timing_items = [
        (
            "ONNX Runtime inference per window, median (clean run)",
            f"{timing['inference_ms_median']:.3f} ms"
            if timing["inference_ms_median"] is not None
            else "n/a",
        ),
        (
            "ONNX Runtime inference per window, max (clean run)",
            f"{timing['inference_ms_max']:.3f} ms"
            if timing["inference_ms_max"] is not None
            else "n/a",
        ),
        ("Windows", timing["windows_timed"]),
        ("Whole clean stream, wall time", f"{timing['stream_wall_ms']:.1f} ms"),
        ("Note", timing["note"] + "; the protocol benchmark (kw10-latency-v1) is in RESULTS.md"),
    ]
    sections.append("<h2>Timing (this run)</h2>" + _dl(timing_items))
    checkpoint = model["checkpoint"]
    runtime = model["runtime"]
    sections.append(
        "<h2>Model and provenance</h2>"
        + _dl(
            [
                ("ONNX model SHA-256", model["onnx_sha256"]),
                ("Export run", model["export_run_id"]),
                ("Export report SHA-256 (pinned)", model["export_report_sha256"]),
                ("Export parity passed (kw10-onnx-v1)", model["export_parity_passed"]),
                ("Checkpoint SHA-256", checkpoint["sha256"]),
                (
                    "Checkpoint",
                    f"run {checkpoint['run_id']}, epoch {checkpoint['epoch']}, "
                    f"{checkpoint['kind']}",
                ),
                ("Labels (order)", ", ".join(labels)),
                ("ONNX opset", model["opset"]),
                ("Feature fingerprint", model["feature_fingerprint"]),
                (
                    "Runtime",
                    f"onnxruntime {runtime['onnxruntime']}, providers {runtime['providers']}, "
                    f"{runtime['intra_op_threads']} intra-op thread(s)",
                ),
                (
                    "Demo code revision",
                    source.get("commit", source.get("unavailable"))
                    + (" (dirty)" if source.get("dirty") else ""),
                ),
            ]
        )
    )
    dataset = record["dataset"]
    sections.append(
        "<h2>Training data</h2>"
        + _dl(
            [
                ("Dataset", f"{dataset['name']} {dataset['version']} ({dataset['subset']})"),
                ("Manifest SHA-256", dataset["manifest_sha256"]),
                ("License", dataset["license"]),
                ("Attribution", dataset["attribution"]),
                ("Citation", dataset["citation"]),
            ]
        )
    )
    if corruption:
        sections.append(
            "<h2>Corruption</h2>"
            + _dl(
                [
                    ("Protocol", corruption["protocol"]),
                    (
                        "Corruption",
                        f"{corruption['name']} v{corruption['version']}: "
                        f"{corruption['description']}",
                    ),
                    ("Severity", f"{corruption['severity']} {corruption['params']}"),
                    ("Seed", f"{corruption['seed']} (from {corruption['identity']})"),
                    ("Applied to", corruption["applied_to"]),
                    ("Measures", json.dumps(corruption["measures"], sort_keys=True)),
                ]
            )
        )
    sections.append(
        "<h2>Streaming contract and configuration</h2>"
        f"<pre>{_e(json.dumps(record['streaming'], indent=2, sort_keys=True))}</pre>"
        f"<pre>{_e(json.dumps(record['config'], indent=2, sort_keys=True))}</pre>"
    )
    sections.append(
        "<h2>Limitations</h2><ul>"
        + "".join(f"<li>{_e(item)}</li>" for item in record["limitations"])
        + "</ul>"
    )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>KWS streaming demo</title>"
        f"<style>{_CSS}</style></head><body><main>" + "".join(sections) + "</main></body></html>\n"
    )
