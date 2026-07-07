#!/bin/bash
# 全量 Soft-PQ 在线测试：精确对照 CSV "Ours" 配置
# 使用 GPU 0-3 并行，覆盖 dinov2 L/G 全部 block、全部码本配置、cls+seg
set -uo pipefail

COFAI_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS=4
OUTPUT_BASE="${OUTPUT_BASE:-$COFAI_ROOT/eval_results}"

export PROJECT_ROOT="$COFAI_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

mkdir -p "$OUTPUT_BASE"

codec_tag_from_ckpt() {
    local name
    name=$(basename "$1" .pt)
    if [[ "$name" =~ _K([0-9]+)_emb([0-9]+)_ ]]; then
        echo "K${BASH_REMATCH[1]}e${BASH_REMATCH[2]}"
    else
        echo "$name"
    fi
}

dataset_from_task() {
    case "$1" in
        cls) echo "imagenet-sel500" ;;
        seg) echo "voc2012-sel100" ;;
    esac
}

slot_from_layer() {
    case "$1" in
        blk05) echo "slot06" ;;
        blk10) echo "slot11" ;;
        blk15) echo "slot16" ;;
        blk20) echo "slot21" ;;
        blk09) echo "slot10" ;;
        blk19) echo "slot20" ;;
        blk29) echo "slot30" ;;
        *) echo "slot00" ;;
    esac
}

result_subdir_from_job() {
    local bb=$1 layer=$2 task=$3 ckpt=$4
    local dataset bb_path slot codec_tag
    dataset=$(dataset_from_task "$task")
    if [ "$task" = "seg" ]; then
        bb_path="dinov2-${bb}-slide"
    else
        bb_path="dinov2-${bb}"
    fi
    slot=$(slot_from_layer "$layer")
    codec_tag=$(codec_tag_from_ckpt "$ckpt")
    echo "${OUTPUT_BASE}/SoftPQ/${dataset}/${bb_path}/${slot}/${codec_tag}"
}

# ================================================================
# Job definitions: "backbone layer task checkpoint_filename"
# 精确对照4个CSV的Ours部分
# ================================================================
JOBS=()

W_L="weights/orfc_2446/dinov2_vitl14_ori"
W_G="weights/orfc_2446/dinov2_vitg14_ori"

# ============== dinov2_vitl14 cls (对照 dinov2_vitl14_cls.csv Ours) ==============
# blk05
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K4_emb32_bt1024_ws_lmbda0.0_tau0.5_lr0.0005_ep300_n5000_s42.pt")
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 cls ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk10
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 cls ${W_L}/blk10_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk15
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 cls ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk20
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K4_emb32_bt1024_ws_tau0.5_lr0.0005_ep300_n5000_s42.pt")
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K32_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 cls ${W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")

# ============== dinov2_vitl14 seg (对照 dinov2_vitl14_seg.csv Ours — 同cls配置) ==============
# blk05
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K4_emb32_bt1024_ws_lmbda0.0_tau0.5_lr0.0005_ep300_n5000_s42.pt")
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk05 seg ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk10
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk10 seg ${W_L}/blk10_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk15
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk15 seg ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk20
JOBS+=("vitl14 blk20 seg ${W_L}/blk20_K4_emb32_bt1024_ws_tau0.5_lr0.0005_ep300_n5000_s42.pt")
JOBS+=("vitl14 blk20 seg ${W_L}/blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 seg ${W_L}/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 seg ${W_L}/blk20_K32_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 seg ${W_L}/blk20_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 seg ${W_L}/blk20_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitl14 blk20 seg ${W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")

# ============== dinov2_vitg14 cls (对照 dinov2_vitg14_cls.csv Ours) ==============
# blk09
JOBS+=("vitg14 blk09 cls ${W_G}/blk09_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 cls ${W_G}/blk09_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 cls ${W_G}/blk09_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 cls ${W_G}/blk09_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 cls ${W_G}/blk09_K256_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 cls ${W_G}/blk09_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk19
JOBS+=("vitg14 blk19 cls ${W_G}/blk19_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 cls ${W_G}/blk19_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 cls ${W_G}/blk19_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 cls ${W_G}/blk19_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 cls ${W_G}/blk19_K256_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 cls ${W_G}/blk19_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk29
JOBS+=("vitg14 blk29 cls ${W_G}/blk29_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 cls ${W_G}/blk29_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 cls ${W_G}/blk29_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 cls ${W_G}/blk29_K64_emb32_bt1536_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 cls ${W_G}/blk29_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 cls ${W_G}/blk29_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")

# ============== dinov2_vitg14 seg (对照 dinov2_vitg14_seg.csv Ours) ==============
# blk09 — K=64/256 用λ=0版本（与cls不同）
JOBS+=("vitg14 blk09 seg ${W_G}/blk09_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 seg ${W_G}/blk09_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 seg ${W_G}/blk09_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 seg ${W_G}/blk09_K64_emb32_bt1536_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 seg ${W_G}/blk09_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk09 seg ${W_G}/blk09_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk19 — K=64/256 用λ=0版本
JOBS+=("vitg14 blk19 seg ${W_G}/blk19_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 seg ${W_G}/blk19_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 seg ${W_G}/blk19_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 seg ${W_G}/blk19_K64_emb32_bt1536_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 seg ${W_G}/blk19_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk19 seg ${W_G}/blk19_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
# blk29 — K=64用λ=0.5（与cls的λ=0不同！seg用λ=0.5）, K=256用λ=0 lr=5e-4
JOBS+=("vitg14 blk29 seg ${W_G}/blk29_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 seg ${W_G}/blk29_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 seg ${W_G}/blk29_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 seg ${W_G}/blk29_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 seg ${W_G}/blk29_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt")
JOBS+=("vitg14 blk29 seg ${W_G}/blk29_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt")

TOTAL=${#JOBS[@]}

# ================================================================
# 检查所有 checkpoint 存在
# ================================================================
echo "============================================================"
echo "  Soft-PQ Full Evaluation (对照CSV Ours配置)"
echo "  Total jobs: $TOTAL"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS parallel)"
echo "  Output: $OUTPUT_BASE"
echo "  Started: $(date)"
echo "============================================================"

MISSING=0
for ((i=0; i<TOTAL; i++)); do
    read -r bb layer task ckpt <<< "${JOBS[$i]}"
    if [ ! -f "$COFAI_ROOT/$ckpt" ]; then
        echo "  MISSING: $ckpt"
        MISSING=$((MISSING+1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo "ERROR: $MISSING checkpoints not found!"
    exit 1
fi
echo "  All $TOTAL checkpoints verified."
echo ""

# ================================================================
# Plan YAML 映射
# ================================================================
get_plan() {
    local bb=$1 layer=$2 task=$3
    local slot=""
    case "$layer" in
        blk05) slot="slot06" ;; blk10) slot="slot11" ;;
        blk15) slot="slot16" ;; blk20) slot="slot21" ;;
        blk09) slot="slot10" ;; blk19) slot="slot20" ;;
        blk29) slot="slot30" ;;
    esac

    if [ "$task" = "cls" ]; then
        echo "conf/plan/dinov2/imagenet-sel500__dinov2-${bb}-${slot}__SoftPQ__cls.yaml"
    else
        echo "conf/plan/dinov2/voc2012-sel100__dinov2-${bb}-slide-${slot}__SoftPQ__semseg.yaml"
    fi
}

# ================================================================
# GPU 并行调度
# ================================================================
declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

run_job() {
    local gpu_idx=$1 bb=$2 layer=$3 task=$4 ckpt=$5
    local gpu_id=${GPUS[$gpu_idx]}
    local plan=$(get_plan "$bb" "$layer" "$task")
    local codec_tag result_subdir log_dir log
    codec_tag=$(codec_tag_from_ckpt "$ckpt")
    result_subdir=$(result_subdir_from_job "$bb" "$layer" "$task" "$ckpt")
    log_dir="$(dirname "$result_subdir")/logs"
    log="${log_dir}/${codec_tag}.log"

    # Backbone local weights path
    local backbone_ckpt=""
    case "$bb" in
        vitl14) backbone_ckpt="${COFAI_ROOT}/weights/dinov2/backbone/dinov2_vitl14_pretrain.pth" ;;
        vitg14) backbone_ckpt="${COFAI_ROOT}/weights/dinov2/backbone/dinov2_vitg14_pretrain.pth" ;;
    esac

    mkdir -p "$log_dir"

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON -m cofai.engine.run_eval \
        "$plan" \
        args.cuda=true \
        args.multi_run=false \
        "model.dino_backbone.ckpt_path=${backbone_ckpt}" \
        "model.dino_codec.codec_path=${COFAI_ROOT}/${ckpt}" \
        "args.result_subdir=${result_subdir}" \
        > "$log" 2>&1
}

wait_for_gpu() {
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            local pid=${GPU_PIDS[$g]}
            if [ "$pid" -eq 0 ]; then
                echo $g
                return
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
                GPU_PIDS[$g]=0
                echo $g
                return
            fi
        done
        sleep 2
    done
}

for ((i=0; i<TOTAL; i++)); do
    read -r bb layer task ckpt <<< "${JOBS[$i]}"
    ckpt_short=$(basename "$ckpt" .pt | cut -c1-40)
    gpu_idx=$(wait_for_gpu)
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU${GPUS[$gpu_idx]}: ${bb}/${layer}/${task} ${ckpt_short}..."
    run_job "$gpu_idx" "$bb" "$layer" "$task" "$ckpt" &
    GPU_PIDS[$gpu_idx]=$!
done

# Wait for remaining
for ((g=0; g<NUM_GPUS; g++)); do
    local_pid=${GPU_PIDS[$g]}
    if [ "$local_pid" -ne 0 ]; then
        wait "$local_pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
    fi
done

echo ""
echo "============================================================"
echo "  Full Evaluation Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Results: $OUTPUT_BASE"
echo "  Finished: $(date)"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "  Check failed logs under: $OUTPUT_BASE/SoftPQ/**/logs/"
fi
