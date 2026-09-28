"""Build the GitHub Pages site from the static showcase page ``docs/index.html``.

Usage (the Pages workflow runs this; standard library only)::

    python scripts/build_pages.py --repo-url https://github.com/OWNER/REPO --ref SHA --out _site

The page links to repository files with relative paths, so it also works when
opened from a local checkout. Pages publishes only the page, so every relative
link to a repository file is rewritten to ``<repo-url>/blob/<ref>/<path>``
(``tree`` for a directory), pinned to the published commit. Fragment-only
links and absolute URLs are kept. A link to a missing file, or one that leaves
the repository, fails the build. The page must not load scripts or external
resources. Writes ``<out>/index.html`` and ``<out>/.nojekyll``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
_HREF = re.compile(r'href="([^"]*)"')
_FORBIDDEN = re.compile(r"<script|<iframe|<link\b|\bsrc=\"(?!data:)", re.IGNORECASE)


class PageError(ValueError):
    """The page has a broken or unsafe link, or loads an external resource."""


def rewrite_links(page: str, *, repo_url: str, ref: str, page_dir: Path, root: Path) -> str:
    """Absolute, commit-pinned repository URLs for every relative link in ``page``."""
    if _FORBIDDEN.search(page):
        raise PageError("the showcase must not contain scripts, frames, or external resources")
    base = repo_url.rstrip("/")
    root = root.resolve()

    def fix(match: re.Match[str]) -> str:
        target = match.group(1)
        if not target or target.startswith("#") or urlsplit(target).scheme:
            return match.group(0)
        path, _, fragment = target.partition("#")
        resolved = (page_dir / path).resolve()
        if resolved != root and root not in resolved.parents:
            raise PageError(f"link leaves the repository: {target}")
        if not resolved.exists():
            raise PageError(f"broken link: {target}")
        kind = "tree" if resolved.is_dir() else "blob"
        relative = resolved.relative_to(root).as_posix()
        url = f"{base}/{kind}/{ref}/{relative}" + (f"#{fragment}" if fragment else "")
        return f'href="{url}"'

    return _HREF.sub(fix, page)


def build(out: Path, *, repo_url: str, ref: str, page: Path = PAGE, root: Path = ROOT) -> Path:
    html = rewrite_links(
        page.read_text(encoding="utf-8"),
        repo_url=repo_url,
        ref=ref,
        page_dir=page.parent,
        root=root,
    )
    out.mkdir(parents=True, exist_ok=True)
    target = out / "index.html"
    target.write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the GitHub Pages showcase site.")
    parser.add_argument("--repo-url", required=True, help="e.g. https://github.com/OWNER/REPO")
    parser.add_argument("--ref", required=True, help="commit SHA (or branch) links point to")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        target = build(args.out, repo_url=args.repo_url, ref=args.ref)
    except PageError as exc:
        print(f"pages build failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
