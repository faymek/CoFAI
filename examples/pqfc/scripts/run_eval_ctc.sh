#!/usr/bin/env bash
# Run PQFC DINOv3 CTC plans (ADE20K semseg / NYUv2 depth), multi-GPU by quality.
#
# VARIANT=transform|noR|both   (default transform)
# TASK=semseg|depth|both
# GPU_IDS=2,4,5,6,7,0,1
# PARALLEL=qualities (default) | plans
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_env.sh"

PLAN_DIR="$SOURCE_ROOT/examples/pqfc/plan/dinov3"
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_ROOT/logs/pqfc/dinov3}"
GPU_IDS="${GPU_IDS:-${GPUS:-0,1}}"
TASK="${TASK:-both}"
VARIANT="${VARIANT:-transform}"
PARALLEL="${PARALLEL:-qualities}"
DRY_RUN="${DRY_RUN:-0}"

# Worktree has no .env; data/backbone live in the CoFAI PROJECT_ROOT checkout.
if [[ ! -d "$PROJECT_ROOT/data/ADEChallengeData2016" && -d "$(dirname "$SOURCE_ROOT")/CoFAI/data/ADEChallengeData2016" ]]; then
  export PROJECT_ROOT="$(dirname "$SOURCE_ROOT")/CoFAI"
fi
if [[ -x "$PROJECT_ROOT/.venv/bin/python" && "${PYTHON:-}" == *"poetry"* ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi
export PYTHON
export OUTPUT_DIR GPU_IDS TASK VARIANT

case "$TASK" in
    semseg|depth|both) ;;
    *)
        echo "ERROR: TASK must be semseg, depth, or both; got $TASK" >&2
        exit 1
        ;;
esac
case "$VARIANT" in
    transform|noR|both) ;;
    *)
        echo "ERROR: VARIANT must be transform, noR, or both; got $VARIANT" >&2
        exit 1
        ;;
esac

echo "SOURCE_ROOT=$SOURCE_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VARIANT=$VARIANT TASK=$TASK GPU_IDS=$GPU_IDS PARALLEL=$PARALLEL"
echo "OUTPUT_DIR=$OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: $PYTHON $SCRIPT_DIR/run_eval_parallel.py"
    exit 0
fi

if [[ "$PARALLEL" == "qualities" ]]; then
    exec $PYTHON "$SCRIPT_DIR/run_eval_parallel.py"
fi

# Fallback: one GPU per plan (qualities sequential inside cofai-eval).
case "$TASK" in
    semseg) TASK_GLOB="*__semseg.yaml" ;;
    depth) TASK_GLOB="*__depth.yaml" ;;
    both) TASK_GLOB="*.yaml" ;;
esac
case "$VARIANT" in
    transform) NAME_FILTER='__PQFC__' ;;
    noR) NAME_FILTER='__PQFC-noR__' ;;
    both) NAME_FILTER='' ;;
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
    echo "ERROR: No PQFC DINOv3 plans matched VARIANT=$VARIANT TASK=$TASK" >&2
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
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    PYTHONPATH="$SOURCE_ROOT" PROJECT_ROOT="$PROJECT_ROOT" \
    $PYTHON -m cofai.engine.run_eval \
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
exit "$failed"
