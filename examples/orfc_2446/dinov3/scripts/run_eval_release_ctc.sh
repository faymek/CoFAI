#!/usr/bin/env bash
# Run the released DINOv3 ORFC-2446 plans through the standard CoFAI entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COFAI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

PLAN_DIR="$COFAI_ROOT/examples/orfc_2446/plan/dinov3"
OUTPUT_DIR="${OUTPUT_DIR:-$COFAI_ROOT/logs/orfc_2446/dinov3}"
GPU_IDS="${GPU_IDS:-${GPUS:-0,1}}"
TASK="${TASK:-both}"
DRY_RUN="${DRY_RUN:-0}"

case "$TASK" in
    semseg)
        PATTERN="*__semseg.yaml"
        ;;
    depth)
        PATTERN="*__depth.yaml"
        ;;
    both)
        PATTERN="*.yaml"
        ;;
    *)
        echo "ERROR: TASK must be semseg, depth, or both; got $TASK" >&2
        exit 1
        ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
mapfile -t PLANS < <(find "$PLAN_DIR" -maxdepth 1 -type f -name "$PATTERN" | sort)
if (( ${#PLANS[@]} == 0 )); then
    echo "ERROR: No DINOv3 ORFC-2446 plans found in $PLAN_DIR." >&2
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
