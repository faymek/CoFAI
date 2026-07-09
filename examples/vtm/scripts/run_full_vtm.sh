#!/usr/bin/env bash
# Full offline VTM: ADE20K val semseg (2000) then NYUv2 test depth (654).
# Run inside tmux to survive SSH disconnect, e.g.:
#   tmux attach -t zlt4
#   cd $PROJECT_ROOT && bash examples/vtm/scripts/run_full_vtm.sh 2>&1 | tee examples/vtm/results/run_full_vtm.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PROJECT_ROOT="$ROOT"
cd "$ROOT"

export GPU="${GPU:-0}"
export WORKERS="${WORKERS:-24}"
export QPS="${QPS:-22 32 42}"

LOG_DIR="$ROOT/examples/vtm/results"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  DINOv3 slot24 offline VTM — full pipeline"
echo "  GPU (extract/replay): $GPU | VTM workers: $WORKERS (CPU)"
echo "  Started: $(date)"
echo "============================================================"

bash examples/vtm/scripts/run_seg_vtm.sh 2>&1 | tee "$LOG_DIR/run_seg_vtm.log"

echo ""
echo "========== seg done, starting depth =========="
echo ""

bash examples/vtm/scripts/run_depth_vtm.sh 2>&1 | tee "$LOG_DIR/run_depth_vtm.log"

echo ""
echo "============================================================"
echo "  Full pipeline DONE: $(date)"
echo "============================================================"
