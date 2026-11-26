#!/usr/bin/env bash
# Evaluate one K2048e8 artifact on ADE or NYU with actual rANS streams.
set -euo pipefail
SOURCE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$SOURCE_ROOT}"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
domain="${1:?usage: eval_k2048.sh ade\|nyu ARTIFACT NAME [Hydra overrides]}"
artifact="${2:?missing artifact}"
run_name="${3:?missing run name}"
shift 3
case "$domain" in
  ade) plan=ade20k-val__dinov3-vitl16-slot24__PQFCv2-K2048e8-l100__semseg ;;
  nyu) plan=nyuv2-val__dinov3-vitl16-slot24__PQFCv2-K2048e8-l100__depth ;;
  *) echo "unknown domain: $domain" >&2; exit 2 ;;
esac
output="${EVAL_OUTPUT:-$PROJECT_ROOT/logs/pqfcv2/k2048_8h_eval}"
mkdir -p "$output"
# Evaluation queues may share a GPU; hold the lock through the whole run.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  exec 9>"$output/.gpu_${CUDA_VISIBLE_DEVICES//,/_}.lock"
  flock 9
fi
"${PYTHON_BIN:-python}" -m cofai.engine.run_eval \
  --config-dir="$SOURCE_ROOT/examples/pqfcv2/plan" --config-name="$plan" \
  "name=${run_name}__${domain}" "args.output_dir=$output" "args.real=true" \
  "model.dino_codec.opq_path=$artifact" "$@"
