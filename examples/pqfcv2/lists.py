"""ADE and NYU image lists shared by PQFCv2 feature extraction."""

from __future__ import annotations

from pathlib import Path


def read_ade_names(list_path: Path) -> list[str]:
    names = []
    with open(list_path, encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                names.append(Path(token).name)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate ADE names in {list_path}")
    return names


def read_nyu_relpaths(list_path: Path) -> list[str]:
    rels = []
    with open(list_path, encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                rels.append(token.lstrip("/"))
    if len(rels) != len(set(rels)):
        raise ValueError(f"Duplicate NYU paths in {list_path}")
    return rels


def nyu_feature_name(relpath: str) -> str:
    return Path(relpath).as_posix().replace("/", "__")


def ade_feature_name(filename: str) -> str:
    return Path(filename).stem + ".npy"


def iter_sources(
    *,
    ade_list: Path,
    ade_root: Path,
    nyu_list: Path,
    nyu_root: Path,
    ade_out: Path,
    nyu_out: Path,
) -> list[tuple[str, Path, Path]]:
    """Return (dataset, image path, feature path) in a stable order."""
    samples: list[tuple[str, Path, Path]] = []
    for name in read_ade_names(ade_list):
        samples.append(("ade", ade_root / name, ade_out / ade_feature_name(name)))
    for rel in read_nyu_relpaths(nyu_list):
        samples.append(("nyu", nyu_root / rel, nyu_out / (Path(nyu_feature_name(rel)).stem + ".npy")))
    names = [path.name for _, _, path in samples]
    if len(names) != len(set(names)):
        raise ValueError("ADE and NYU feature filenames collide")
    return samples
