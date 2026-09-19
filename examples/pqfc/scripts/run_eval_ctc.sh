#!/usr/bin/env bash
# Run PQFC DINOv3 CTC plans (ADE20K semseg / NYUv2 depth).
#
# VARIANT=transform  (default)  — with orthogonal R; uses ORFC-2446 release npz
# VARIANT=noR                   — no transform; placeholder weights (will fail until filled)
# VARIANT=both
# TASK=semseg|depth|both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COFAI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PLAN_DIR="$COFAI_ROOT/examples/pqfc/plan/dinov3"
OUTPUT_DIR="${OUTPUT_DIR:-$COFAI_ROOT/logs/pqfc/dinov3}"
GPU_IDS="${GPU_IDS:-${GPUS:-0,1}}"
TASK="${TASK:-both}"
VARIANT="${VARIANT:-transform}"
DRY_RUN="${DRY_RUN:-0}"

case "$TASK" in
    semseg) TASK_GLOB="*__semseg.yaml" ;;
    depth) TASK_GLOB="*__depth.yaml" ;;
    both) TASK_GLOB="*.yaml" ;;
    *)
        echo "ERROR: TASK must be semseg, depth, or both; got $TASK" >&2
        exit 1
        ;;
esac

case "$VARIANT" in
    transform) NAME_FILTER='__PQFC__' ;;
    noR) NAME_FILTER='__PQFC-noR__' ;;
    both) NAME_FILTER='' ;;
    *)
        echo "ERROR: VARIANT must be transform, noR, or both; got $VARIANT" >&2
        exit 1
        ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
mapfile -t ALL_PLANS < <(find "$PLAN_DIR" -maxdepth 1 -type f -name "$TASK_GLOB" | sort)
PLANS=()
for plan_path in "${ALL_PLANS[@]}"; do
    base="$(basename "$plan_path")"
    if [[ -z "$NAME_FILTER" ]]; then
        PLANS+=("$plan_path")
    elif [[ "$NAME_FILTER" == '__PQFC__' && "$base" == *"__PQFC__"* && "$base" != *"__PQFC-noR__"* ]]; then
        PLANS+=("$plan_path")
    elif [[ "$NAME_FILTER" == '__PQFC-noR__' && "$base" == *"__PQFC-noR__"* ]]; then
        PLANS+=("$plan_path")
    fi
done

if (( ${#PLANS[@]} == 0 )); then
    echo "ERROR: No PQFC DINOv3 plans matched VARIANT=$VARIANT TASK=$TASK in $PLAN_DIR." >&2
    exit 1
fi

LOG_DIR="$OUTPUT_DIR/runner_logs"
mkdir -p "$LOG_DIR"

run_plan() {
    local gpu_id=$1
    local plan_path=$2
    local plan_name
    plan_name="$(basename "$plan_path" .yaml)"
    local log_file="$LOG_DIR/$plan_name.log"

    echo "[gpu=$gpu_id] $plan_name"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "CUDA_VISIBLE_DEVICES=$gpu_id poetry -C $COFAI_ROOT run cofai-eval $plan_path args.multi_run=true args.cuda=true args.real=true args.output_dir=$OUTPUT_DIR" \
            >"$log_file"
        return
    fi
    CUDA_VISIBLE_DEVICES="$gpu_id" poetry -C "$COFAI_ROOT" run cofai-eval \
        "$plan_path" \
        args.multi_run=true \
        args.cuda=true \
        args.real=true \
        "args.output_dir=$OUTPUT_DIR" \
        >"$log_file" 2>&1
}

pids=()
labels=()
for index in "${!PLANS[@]}"; do
    gpu="${GPUS[index % ${#GPUS[@]}]}"
    run_plan "$gpu" "${PLANS[index]}" &
    pids+=("$!")
    labels+=("${PLANS[index]}")
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        echo "FAIL: ${labels[index]}" >&2
        failed=$((failed + 1))
    fi
done

echo "Completed plans: $((${#PLANS[@]} - failed)) / ${#PLANS[@]}"
if (( failed > 0 )); then
    exit 1
fi
