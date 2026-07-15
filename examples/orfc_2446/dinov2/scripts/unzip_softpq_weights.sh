#!/bin/bash
# Unzip SoftPQ release archives into weights/orfc_2446/{dinov2_vitl14_ori,dinov2_vitg14_ori}/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORFC2446_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$(dirname "$ORFC2446_DIR")")")"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$COFAI_ROOT/weights/orfc_2446}"

cd "$COFAI_ROOT"

for sub in dinov2_vitl14_ori dinov2_vitg14_ori; do
    zip="$WEIGHTS_ROOT/${sub}.zip"
    dest="$WEIGHTS_ROOT/$sub"
    if [ ! -f "$zip" ]; then
        echo "Missing $zip — run cofai-download first."
        exit 1
    fi
    if [ -L "$dest" ]; then
        echo "Removing symlink $dest"
        rm -f "$dest"
    elif [ -d "$dest" ]; then
        echo "Refreshing $dest from zip"
        rm -rf "$dest"
    fi
    echo "Unzipping $zip -> $WEIGHTS_ROOT/"
    unzip -qo "$zip" -d "$WEIGHTS_ROOT"
    n=$(find "$dest" -maxdepth 1 -name '*.npz' | wc -l)
    echo "  $sub: $n .npz files"
done

echo "Done. SoftPQ weights ready under $WEIGHTS_ROOT/"
