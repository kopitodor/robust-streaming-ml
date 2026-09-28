"""The public showcase page and LinkedIn copy stay consistent with RESULTS.md and publishable."""

from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
RESULTS = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
BUILD: dict[str, Any] = runpy.run_path(str(ROOT / "scripts" / "build_pages.py"))


def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split(" | ")]


def results_rows(prefix: str) -> list[list[str]]:
    return [cells(line) for line in RESULTS.splitlines() if line.startswith(prefix)]


def page_row(attribute: str, key: str) -> list[str]:
    match = re.search(rf'<tr[^>]*{attribute}="{re.escape(key)}"[^>]*>(.*?)</tr>', page(), re.S)
    assert match, f"missing row {key}"
    return [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", match[1])]


def test_robustness_table_matches_results() -> None:
    rows = results_rows(
        "| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) |"
    )
    assert len(rows) == 15
    section = RESULTS[RESULTS.index("## Robustness (Milestone 5)") :]
    for row in rows:
        corruption, severity = row[2].removesuffix(" v1"), row[3]
        got = page_row("data-condition", f"{corruption}/{severity}")
        assert got[:3] == [corruption, severity, row[5]]
        assert (got[4], got[5]) == (row[6], row[7])  # degradation, macro F1
        nll_ece = re.search(
            rf"^\| {corruption} \| {re.escape(severity)} \| ([0-9.]+) \| \S+ \| ([0-9.]+) \|",
            section,
            re.M,
        )
        assert nll_ece and (got[6], got[7]) == (nll_ece[1], nll_ece[2])
    assert page().count("data-condition=") == 15


def test_latency_and_clean_tables_match_results() -> None:
    for row in results_rows("| `20260928T004340Z_kw10-onnx-latency_1df4024f` |"):
        if row[1].startswith("PyTorch"):
            path = "PyTorch eager streaming step (reference)"
        elif "session.run" in row[11]:
            path = "ONNX session.run alone"
        else:
            path = "ONNX streaming step"
        threads = row[4].split()[0]
        got = page_row("data-latency", f"{path}/{threads}")
        assert got[1:] == [threads, row[9], row[10]]
        if "Primary" in row[11]:
            assert f"{float(row[9]):.2f} ms" in page() and f"p95 {float(row[10]):.2f} ms" in page()
    for row in results_rows("| `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` |"):
        values = f"<td>{row[5]}</td>"
        assert values in page()
        for value in row[6:10]:
            assert f'<td class="num">{value}</td>' in page()


@pytest.mark.parametrize(
    "fact",
    ["1.507e-4", "1.764e-4", "3.61e-5", "35,629", "4,074 / 4,074", "0.4639", "0.9659", "3,935"],
)
def test_quoted_facts_appear_in_results(fact: str) -> None:
    assert fact in page() and fact in RESULTS


def test_page_is_static_scoped_and_private() -> None:
    text = page()
    assert not re.search(r"<script|<iframe|<img|<link\b|\bsrc=", text, re.I)
    # no local paths or personal email addresses
    for forbidden in ("C:\\", "C:/Users", "/home/", "/Users/", "AppData", "@gmail.com"):
        assert forbidden not in text
    lowered = text.lower()
    assert "not general speech recognition" in lowered and "not production software" in lowered
    assert "no real-time or state-of-the-art claim" in lowered


def test_pages_build_pins_every_repository_link(tmp_path: Path) -> None:
    out = BUILD["build"](tmp_path / "site", repo_url="https://github.com/owner/repo", ref="abc123")
    built = out.read_text(encoding="utf-8")
    assert (tmp_path / "site" / ".nojekyll").is_file()
    hrefs = re.findall(r'href="([^"]*)"', built)
    assert hrefs and all(
        h.startswith(("#", "https://github.com/owner/repo/blob/abc123/", "https://")) for h in hrefs
    )
    assert "https://github.com/owner/repo/blob/abc123/RESULTS.md" in hrefs
    assert "https://github.com/owner/repo/blob/abc123/docs/evaluation-protocol.md" in hrefs
    assert "https://github.com/owner/repo/blob/abc123/README.md#reproducing-the-pipeline" in hrefs
    assert "https://creativecommons.org/licenses/by/4.0/" in hrefs


@pytest.mark.parametrize(
    ("html", "match"),
    [
        ('<a href="missing.md">x</a>', "broken link"),
        ('<a href="../../outside.md">x</a>', "leaves the repository"),
        ('<script src="https://example.com/x.js"></script>', "must not contain"),
        ('<img src="https://example.com/x.png">', "must not contain"),
    ],
)
def test_pages_build_rejects_unsafe_pages(tmp_path: Path, html: str, match: str) -> None:
    source = tmp_path / "repo" / "docs" / "index.html"
    source.parent.mkdir(parents=True)
    source.write_text(html, encoding="utf-8")
    with pytest.raises(BUILD["PageError"], match=match):
        BUILD["build"](
            tmp_path / "out", repo_url="https://x", ref="r", page=source, root=source.parents[1]
        )


def test_linkedin_bullets_follow_the_requested_format() -> None:
    text = (ROOT / "docs" / "linkedin.md").read_text(encoding="utf-8")
    section = text.split("## Achievement bullets", 1)[1].split("\n## ", 1)[0]
    bullets = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(bullets) == 3
    for bullet in bullets:
        assert re.fullmatch(r"- Achieved .+?, measured by .+?, by [a-z]+ing .+\.", bullet), bullet
    for fact in ("0.9639", "4,074", "61,052", "0.4639", "0.9659", "1.8e-4", "39,703", "2.43 ms"):
        assert fact in text and fact in RESULTS + (ROOT / "README.md").read_text(encoding="utf-8")
