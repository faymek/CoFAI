# DINOv3 slot24 离线 VTM 流水线

对 DINOv3 ViT-L/16 **slot24** 的完整 encoder token `(N, 1024)`（含 CLS+REG+patch）做 VTM 编解码，分阶段 **extract → vtm → replay**，支持 **resume** 与 **fail-fast**。

## 数据集规模（评测集）

| 任务 | 数据集 | 划分 | 样本数 |
|------|--------|------|--------|
| 分割 semseg | ADE20K | validation | 2,000 |
| 深度 depth | NYUv2 | test (`nyu_test.txt`) | 654 |

## 前置

```bash
export PROJECT_ROOT=/path/to/CoFAI
# VTM 二进制默认: ORFC/coding/vtm_baseline/
# DINOv3 backbone + task heads 见 examples/ctc/README-DINOv3.md
```

## 全量评测（建议在 tmux 中运行）

```bash
tmux attach -t zlt4   # 或新建: tmux new -s zlt4
cd $PROJECT_ROOT
export GPU=0 WORKERS=24

# ADE20K val 2000 + NYUv2 test 654，顺序执行
bash examples/vtm/scripts/run_full_vtm.sh 2>&1 | tee examples/vtm/results/run_full_vtm.log

# 或分步：
# bash examples/vtm/scripts/run_seg_vtm.sh
# bash examples/vtm/scripts/run_depth_vtm.sh
```

- **extract / replay**：使用 GPU（`CUDA_VISIBLE_DEVICES=$GPU`）
- **vtm**：仅 CPU（`CUDA_VISIBLE_DEVICES=""`），24 workers，不占用显存
- 各阶段支持 **resume**（已存在文件则跳过）

## 特征目录

```
features/dinov3_vitl16_slot24/
├── semseg/
│   ├── tokens/{stem}.npy      # (N, 1024)
│   ├── meta/{stem}.npz
│   └── decoded/qp{22,32,42}/{stem}.npy
└── depth/                     # 同上
```

## 说明

- 离线 VTM 使用 **per-image min-max linear quant**（与 `compression_vit/run_vtm_plaindetr.py` 一致），与在线 `VtmLatentCodec` 的固定 `trun_*` 策略不同。
- VTM 任一 worker 失败会立即 `exit 1`；修复后可重跑 `vtm` 阶段（resume）。
