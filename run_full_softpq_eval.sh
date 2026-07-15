#!/usr/bin/env bash
# Soft-PQ full online eval — delegates to the ORFC-2446 reproduction script
# (npz weights + long result paths). Do not point at .pt here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/examples/orfc_2446/dinov2/scripts/run_online_eval_all.sh" "$@"
