#!/usr/bin/env bash
# Run every published DINOv2 ORFC plan. The YAML plans are the only job list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COFAI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

PLAN_DIR="$COFAI_ROOT/examples/orfc_2446/plan/dinov2"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-$COFAI_ROOT/logs/orfc_2446/dinov2}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$COFAI_ROOT/$OUTPUT_DIR"
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} == 0 )) || [[ -z "${GPUS[0]}" ]]; then
    echo "ERROR: GPU_IDS must contain at least one GPU id." >&2
    exit 1
fi

mapfile -t PLANS < <(find "$PLAN_DIR" -maxdepth 1 -type f -name '*__ORFC__*.yaml' -printf '%f\n' | sort)
if (( ${#PLANS[@]} == 0 )); then
    echo "ERROR: No DINOv2 ORFC plans found in $PLAN_DIR." >&2
    exit 1
fi

LOG_DIR="$OUTPUT_DIR/runner_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  DINOv2 ORFC plan suite"
echo "  Plans: ${#PLANS[@]}"
echo "  GPUs: $GPU_IDS"
echo "  Output: $OUTPUT_DIR"
echo "  Dry run: $DRY_RUN"
echo "============================================================"

run_plan() {
    local gpu_id=$1
    local plan_name=$2
    local plan_path="$PLAN_DIR/$plan_name"
    local log_file="$LOG_DIR/${plan_name%.yaml}.log"

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

completed=0
failed=0
for ((start=0; start<${#PLANS[@]}; start+=${#GPUS[@]})); do
    pids=()
    labels=()
    for ((offset=0; offset<${#GPUS[@]} && start+offset<${#PLANS[@]}; offset++)); do
        plan_name="${PLANS[start+offset]}"
        run_plan "${GPUS[offset]}" "$plan_name" &
        pids+=("$!")
        labels+=("$plan_name")
    done

    for index in "${!pids[@]}"; do
        if wait "${pids[index]}"; then
            completed=$((completed + 1))
        else
            echo "FAIL: ${labels[index]} (see $LOG_DIR/${labels[index]%.yaml}.log)" >&2
            failed=$((failed + 1))
        fi
    done
done

echo "Completed plans: $completed / ${#PLANS[@]}"
if (( failed > 0 )); then
    exit 1
fi
