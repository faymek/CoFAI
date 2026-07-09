#!/bin/bash
# Full pipeline: smoke (optional) -> patch-only train -> patch replay -> rate sweep -> rate replay.
set -euo pipefail

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
SCRIPT_DIR="examples/orfc_2446_dinov3/scripts"

SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

if [ "$SKIP_SMOKE" != "1" ]; then
  echo "========== SMOKE TEST =========="
  bash "$SCRIPT_DIR/smoke_patch_only.sh"
  if [ "$SMOKE_ONLY" = "1" ]; then
    echo "SMOKE_ONLY=1, exiting."
    exit 0
  fi
fi

echo ""
echo "========== STEP 1: patch-only training =========="
bash "$SCRIPT_DIR/run_train_patch_opq.sh"

echo ""
echo "========== STEP 1 replay =========="
bash "$SCRIPT_DIR/run_replay_patch_opq.sh"

echo ""
echo "========== STEP 2: rate sweep training =========="
bash "$SCRIPT_DIR/run_train_rate_sweep.sh"

echo ""
echo "========== STEP 2 replay =========="
bash "$SCRIPT_DIR/run_replay_rate_sweep.sh"

echo ""
echo "All pipelines finished."
