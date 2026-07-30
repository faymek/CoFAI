#!/usr/bin/env python3
"""Package the eight DINOv3 ORFC artifacts consumed by the formal plans."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from cofai.latent_codecs.orfc import load_orfc_artifact


SOURCE_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(SOURCE_ROOT / ".env")

RELEASE_NAMES = {
    (16, 32): "blk23_K16_e32.npz",
    (256, 32): "blk23_K256_e32.npz",
    (1024, 32): "blk23_K1024_e32.npz",
    (512, 16): "blk23_K512_e16.npz",
    (1024, 16): "blk23_K1024_e16.npz",
    (64, 8): "blk23_K64_e8.npz",
    (256, 8): "blk23_K256_e8.npz",
    (512, 8): "blk23_K512_e8.npz",
}
RELEASE_IDENTITY = {
    "proposal": "ORFC-2446",
    "backbone": "dinov3_vitl16",
    "layer": "blk23",
    "slot": 24,
    "norm_mode": "split_reg_cls_patch",
    "n_prefix": 5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_release_identity(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(RELEASE_IDENTITY).difference(data.files))
        if missing:
            raise KeyError(f"DINOv3 release artifact is missing identity metadata {missing}: {path}")
        actual = {key: data[key].item() if hasattr(data[key], "item") else data[key] for key in RELEASE_IDENTITY}
    mismatches = {key: (actual[key], expected) for key, expected in RELEASE_IDENTITY.items() if actual[key] != expected}
    if mismatches:
        raise ValueError(f"DINOv3 release identity mismatch for {path}: {mismatches}")


def _collect_sources(paths: list[Path]) -> dict[tuple[int, int], Path]:
    candidates: dict[tuple[int, int], list[Path]] = {key: [] for key in RELEASE_NAMES}
    for path in paths:
        artifact = load_orfc_artifact(path)
        if artifact["feature_dim"] != 1024:
            continue
        key = (artifact["entries"], artifact["embedding_dim"])
        if key in candidates:
            _validate_release_identity(path)
            candidates[key].append(path)

    missing = [RELEASE_NAMES[key] for key, values in candidates.items() if not values]
    ambiguous = {
        RELEASE_NAMES[key]: [str(path) for path in values] for key, values in candidates.items() if len(values) > 1
    }
    if missing or ambiguous:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if ambiguous:
            details.append(f"ambiguous={ambiguous}")
        raise ValueError(
            "Release packaging requires exactly one 1024-D artifact for every "
            f"(K, embedding_dim) pair; {'; '.join(details)}"
        )
    return {key: values[0] for key, values in candidates.items()}


def main() -> None:
    project_root_value = os.environ.get("PROJECT_ROOT")
    if not project_root_value:
        raise RuntimeError(f"PROJECT_ROOT is required; set it or add it to {SOURCE_ROOT / '.env'}")
    project_root = Path(project_root_value).expanduser().resolve()
    parser = argparse.ArgumentParser(description="Validate and package the eight ORFC DINOv3 plan artifacts")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=project_root / "weights/orfc_2446/dinov3_vitl16",
        help="Directory scanned for training-exported .npz files",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        default=[],
        help="Explicit source artifact; repeat eight times to avoid scan ambiguity",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "weights/orfc_2446/dinov3_vitl16_ori",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.artifact:
        paths = [path.expanduser().resolve() for path in args.artifact]
    else:
        input_dir = args.input_dir.expanduser().resolve()
        paths = sorted(path for path in input_dir.glob("*.npz") if path.is_file())
    sources = _collect_sources(paths)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = []
    for key, release_name in RELEASE_NAMES.items():
        source = sources[key]
        destination = output_dir / release_name
        source_hash = _sha256(source)
        if destination.exists() and not args.force:
            if _sha256(destination) != source_hash:
                raise FileExistsError(f"{destination} differs from {source}; pass --force to replace it")
            print(f"[keep] {destination}")
        else:
            shutil.copy2(source, destination)
            print(f"[copy] {source} -> {destination}")
        checksums.append(f"{source_hash}  {release_name}")

    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"[ok] {len(RELEASE_NAMES)} artifacts and {checksum_path}")


if __name__ == "__main__":
    main()
