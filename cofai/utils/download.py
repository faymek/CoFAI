"""Batch downloader for files listed in a plain-text manifest.

A *manifest* is a UTF-8 ``.txt`` file where each non-empty, non-comment line
describes one file to place into the repository::

    # <comment lines start with '#'>
    <dest_path>    [<url>]    [<checksum>]

* ``dest_path``  destination path, relative to the repo root (or ``--root``).
                 The helper creates parent directories automatically.
* ``url``        *(optional)* an ``http(s)://`` direct link.  **If omitted, the
                 url defaults to ``<base_url>/<dest_path>``**, where ``base_url``
                 is :data:`DEFAULT_BASE_URL` (override with ``--base-url``).
* ``checksum``   *(optional)* one of ``sha256:<hex>``, ``md5:<hex>`` or
                 ``size:<bytes>`` used to verify the downloaded file.  For
                 ``sha256``/``md5`` a *prefix* of the digest is accepted (e.g.
                 ``sha256:95253fe4``), which keeps manifests short.

Fields are separated by whitespace, so paths/urls must not contain spaces.
A trailing token is treated as a *checksum* when it looks like
``sha256:``/``md5:``/``size:``; otherwise it is treated as the *url*.  So a line
may be as short as just ``<dest_path>`` (url derived from the base) or
``<dest_path>  sha256:..`` (base url + checksum).

Typical use::

    # download everything that is missing / corrupt
    poetry run cofai-download weights/weights.manifest.txt

    # only check, never download
    poetry run cofai-download weights/weights.manifest.txt --verify

    # show what would happen
    poetry run cofai-download weights/weights.manifest.txt --dry-run

The same entry points are available programmatically::

    from cofai.utils.download import download_manifest, verify_manifest
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from tqdm import tqdm


_CHUNK = 1 << 20  # 1 MiB

#: Base URL used to derive a download link when a manifest line omits the url.
#: The final url is ``DEFAULT_BASE_URL.rstrip('/') + '/' + dest_path``.
DEFAULT_BASE_URL = "https://medialab.sjtu.edu.cn/files/CoFAI-share"

_CHECKSUM_KINDS = ("sha256", "md5", "size")


# --------------------------------------------------------------------------- #
# Manifest model
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    """One manifest line: where to put a file and how to verify it."""

    dest: str
    url: str
    algo: Optional[str] = None  # "sha256" | "md5" | "size" | None
    value: Optional[str] = None


class Status(str, Enum):
    OK = "ok"  # present and verified (or present without checksum)
    MISSING = "missing"  # file not on disk
    SIZE_MISMATCH = "size_mismatch"  # size differs from manifest
    CHECKSUM_MISMATCH = "checksum_mismatch"  # hash differs from manifest
    DOWNLOADED = "downloaded"  # freshly fetched (+ verified if checksum given)
    SKIPPED = "skipped"  # already complete, download skipped
    FAILED = "failed"  # network / IO error during download


#: Statuses that make a run "not ok" (used by Summary.ok and the report).
_BAD_STATUSES = (Status.MISSING, Status.SIZE_MISMATCH, Status.CHECKSUM_MISMATCH, Status.FAILED)


@dataclass
class Result:
    entry: Entry
    status: Status
    detail: str = ""


@dataclass
class Summary:
    results: List[Result] = field(default_factory=list)

    def add(self, result: Result) -> None:
        self.results.append(result)

    @property
    def ok(self) -> bool:
        return not any(r.status in _BAD_STATUSES for r in self.results)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_manifest(
    manifest_path: os.PathLike | str,
    base_url: str = DEFAULT_BASE_URL,
) -> List[Entry]:
    """Parse a manifest file into a list of :class:`Entry` objects.

    ``base_url`` is used to derive the download url for any line that omits one.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    base = base_url.rstrip("/")
    entries: List[Entry] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        dest = parts[0]
        url = None
        algo = value = None
        for token in parts[1:]:
            if _is_checksum_token(token):
                if algo is not None:
                    raise ValueError(f"{path}:{lineno}: more than one checksum given: {raw!r}")
                algo, value = _parse_checksum(token, path, lineno)
            else:
                if url is not None:
                    raise ValueError(f"{path}:{lineno}: more than one url given: {raw!r}")
                url = token
        if url is None:  # derive from base url + dest path
            url = f"{base}/{dest.lstrip('/')}"
        entries.append(Entry(dest=dest, url=url, algo=algo, value=value))
    return entries


def _is_checksum_token(token: str) -> bool:
    prefix = token.split(":", 1)[0].lower()
    return ":" in token and prefix in _CHECKSUM_KINDS


def _parse_checksum(token: str, path: Path, lineno: int) -> Tuple[str, str]:
    algo, value = token.split(":", 1)
    algo = algo.lower()
    if algo not in _CHECKSUM_KINDS:
        raise ValueError(f"{path}:{lineno}: unsupported checksum kind {algo!r}")
    return algo, value


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, entry: Entry) -> Tuple[Status, str]:
    """Check a single file on disk against its manifest entry."""
    if not path.exists():
        return Status.MISSING, "not found"
    if entry.algo == "size":
        actual = path.stat().st_size
        expected = int(entry.value)
        if actual != expected:
            return Status.SIZE_MISMATCH, f"size {actual} != expected {expected}"
        return Status.OK, f"size {actual}"
    if entry.algo in {"sha256", "md5"}:
        actual = _hash_file(path, entry.algo)
        expected = (entry.value or "").lower()
        # Allow a short prefix of the digest (e.g. sha256:95253fe4) for brevity.
        actual_cmp = actual[: len(expected)] if 0 < len(expected) < len(actual) else actual
        if actual_cmp != expected:
            return Status.CHECKSUM_MISMATCH, f"{entry.algo} {actual} != expected {entry.value}"
        suffix = " (prefix)" if len(expected) < len(actual) else ""
        return Status.OK, f"{entry.algo} verified{suffix}"
    return Status.OK, f"present ({path.stat().st_size} bytes, no checksum)"


def verify_manifest(
    manifest_path: os.PathLike | str,
    root: os.PathLike | str = ".",
    base_url: str = DEFAULT_BASE_URL,
) -> Summary:
    """Check every entry without downloading anything."""
    root = Path(root)
    summary = Summary()
    for entry in parse_manifest(manifest_path, base_url=base_url):
        status, detail = verify_file(root / entry.dest, entry)
        summary.add(Result(entry=entry, status=status, detail=detail))
    return summary


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def _download_http(url: str, dest: Path, *, quiet: bool = False) -> None:
    """Stream ``url`` to a ``.part`` file (resuming if present) then rename it."""
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True) as resp:
        # If the server ignored the Range request, restart from scratch.
        if existing and resp.status_code != 206:
            existing = 0
            part.unlink(missing_ok=True)
        resp.raise_for_status()
        total = resp.headers.get("Content-Length")
        total = (int(total) + existing) if total is not None else None
        with tqdm(
            total=total, initial=existing, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name[:30], disable=quiet,
        ) as bar, part.open("ab" if existing else "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                f.write(chunk)
                bar.update(len(chunk))

    part.replace(dest)


def download_entry(
    entry: Entry,
    root: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    retries: int = 3,
    quiet: bool = False,
) -> Result:
    """Download a single entry, skipping/redownloading based on verification."""
    dest = root / entry.dest

    if not force:
        status, detail = verify_file(dest, entry)
        if status == Status.OK:
            return Result(entry, Status.SKIPPED, f"already present ({detail})")
        # MISSING / SIZE_MISMATCH / CHECKSUM_MISMATCH -> (re)download below.

    if dry_run:
        reason = "force" if force else "missing/invalid"
        return Result(entry, Status.DOWNLOADED, f"[dry-run] would download ({reason}) -> {entry.dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            _download_http(entry.url, dest, quiet=quiet)
            break
        except Exception as exc:  # noqa: BLE001 - report and retry
            last_err = f"{type(exc).__name__}: {exc}"
            if not quiet:
                print(f"  attempt {attempt}/{retries} failed: {last_err}", file=sys.stderr)
    else:
        return Result(entry, Status.FAILED, last_err)

    # Post-download verification (if a checksum was provided).
    if entry.algo:
        status, detail = verify_file(dest, entry)
        if status != Status.OK:
            return Result(entry, status, f"verification after download failed: {detail}")
    return Result(entry, Status.DOWNLOADED, "ok")


def download_manifest(
    manifest_path: os.PathLike | str,
    root: os.PathLike | str = ".",
    *,
    base_url: str = DEFAULT_BASE_URL,
    force: bool = False,
    dry_run: bool = False,
    retries: int = 3,
    quiet: bool = False,
) -> Summary:
    """Download every entry in a manifest into ``root``."""
    root = Path(root)
    entries = parse_manifest(manifest_path, base_url=base_url)
    summary = Summary()
    for i, entry in enumerate(entries, start=1):
        if not quiet:
            print(f"[{i}/{len(entries)}] {entry.dest}")
        result = download_entry(entry, root, force=force, dry_run=dry_run, retries=retries, quiet=quiet)
        summary.add(result)
        if not quiet:
            print(f"    -> {result.status.value}: {result.detail}")
    return summary


# --------------------------------------------------------------------------- #
# Reporting / CLI
# --------------------------------------------------------------------------- #
def print_report(summary: Summary) -> None:
    counts = Counter(r.status for r in summary.results)
    print("\n=== summary ===")
    for status in Status:
        if counts[status]:
            print(f"  {status.value:>18}: {counts[status]}")
    problems = [r for r in summary.results if r.status in _BAD_STATUSES]
    if problems:
        print("\nproblems:")
        for r in problems:
            print(f"  [{r.status.value}] {r.entry.dest}: {r.detail}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cofai-download",
        description="Batch-download files listed in a TXT manifest into the repo, with verification.",
    )
    parser.add_argument("manifest", help="path to the .txt manifest file")
    parser.add_argument(
        "--root",
        default=os.environ.get("PROJECT_ROOT", "."),
        help="destination root for relative paths (default: $PROJECT_ROOT or cwd)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("COFAI_DOWNLOAD_BASE_URL", DEFAULT_BASE_URL),
        help="base url used when a manifest line omits the url (default: %(default)s)",
    )
    parser.add_argument("--verify", action="store_true", help="only check files, do not download")
    parser.add_argument("--dry-run", action="store_true", help="show what would be downloaded, without downloading")
    parser.add_argument("--force", action="store_true", help="redownload even if files already exist/verify")
    parser.add_argument("--retries", type=int, default=3, help="download attempts per file (default: 3)")
    parser.add_argument("--quiet", action="store_true", help="suppress per-file progress output")
    args = parser.parse_args(argv)

    try:
        if args.verify:
            summary = verify_manifest(args.manifest, root=args.root, base_url=args.base_url)
            if not args.quiet:
                for r in summary.results:
                    print(f"[{r.status.value}] {r.entry.dest}: {r.detail}")
        else:
            summary = download_manifest(
                args.manifest,
                root=args.root,
                base_url=args.base_url,
                force=args.force,
                dry_run=args.dry_run,
                retries=args.retries,
                quiet=args.quiet,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_report(summary)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
