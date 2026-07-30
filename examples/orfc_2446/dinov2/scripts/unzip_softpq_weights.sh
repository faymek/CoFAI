#!/usr/bin/env bash
# Extract the two DINOv2 SoftPQ release archives under the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COFAI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-$COFAI_ROOT/weights/orfc_2446}"

for name in dinov2_vitl14_ori dinov2_vitg14_ori; do
    archive="$WEIGHTS_ROOT/$name.zip"
    destination="$WEIGHTS_ROOT/$name"

    if [[ ! -f "$archive" ]]; then
        echo "ERROR: Missing $archive. Run cofai-download first." >&2
        exit 1
    fi

    if [[ -L "$destination" ]]; then
        rm -f "$destination"
    elif [[ -d "$destination" ]]; then
        rm -rf "$destination"
    fi

    echo "Extracting $archive"
    unzip -qo "$archive" -d "$WEIGHTS_ROOT"
    count=$(find "$destination" -maxdepth 1 -name '*.npz' | wc -l)
    echo "  $name: $count NPZ files"
done
