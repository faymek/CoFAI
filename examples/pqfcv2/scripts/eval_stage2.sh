#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$SOURCE_ROOT}" PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
N="${1:?usage: eval_stage2.sh LATENT_TOKENS [additional Hydra overrides]}"; shift
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/weights/pqfcv2/dinov3/stage2_N$N/best.pt}"
WEIGHTS="${DINOV3_WEIGHTS:-$PROJECT_ROOT/weights/dinov3}"
BACKBONE="${BACKBONE:-$WEIGHTS/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors}"
DEPTH_HEAD="${DEPTH_HEAD:-$WEIGHTS/dpt_head/dinov3_vitl16_depth_nyuv2_linear_head.pth}"
SEMSEG_HEAD="${SEMSEG_HEAD:-$WEIGHTS/semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth}"
NYU="${NYU_ROOT:-$PROJECT_ROOT/data/NYU}"
ADE="${ADE_ROOT:-$PROJECT_ROOT/data/ADEChallengeData2016}"
OUTPUT="${EVAL_OUTPUT:-$PROJECT_ROOT/logs/pqfcv2/stage2_N$N}"
STAGE1="${STAGE1:-$PROJECT_ROOT/weights/pqfcv2/dinov3/stage1_K2048_e8_postln_l100_splitlr/stage1_K2048_e8.npz}"
COMMON=("model.dino_codec.stage1_path=$STAGE1" "args.real=true" "args.output_dir=$OUTPUT" "model.dino_codec.checkpoint_path=$CHECKPOINT" "model.dino_backbone.ckpt_path=$BACKBONE")
"${PYTHON_BIN:-python}" -m cofai.engine.run_eval --config-dir="$SOURCE_ROOT/examples/pqfcv2/plan" --config-name=nyuv2-val__dinov3-vitl16-slot24__PQFCv2-stage2__depth "${COMMON[@]}" "dataset.root=$NYU" "model.heads.depth.checkpoint=$DEPTH_HEAD" "$@"
"${PYTHON_BIN:-python}" -m cofai.engine.run_eval --config-dir="$SOURCE_ROOT/examples/pqfcv2/plan" --config-name=ade20k-val__dinov3-vitl16-slot24__PQFCv2-stage2__semseg "${COMMON[@]}" "dataset.root=$ADE" "model.heads.semseg.checkpoint=$SEMSEG_HEAD" "$@"
