# VTC 评估代码

这个评估系统提供了完整的Visual Token Codec(VTC)模型评估功能，包含两个提案

- AI M2418：Visual Token Codec 双路径 token 编码
    - VTC on DINOv2-reg4
- AI M2448：基于DINOv3的特征编码器
    - VTC on DINOv3

## 下载数据与权重

权重与数据集压缩包均通过下载清单 `dinov3-vtc.manifest.txt` 一键拉取并按 sha256 校验（默认从 `https://medialab.sjtu.edu.cn/files/CoFAI-share/` 按相对路径下载）。在仓库根目录执行：

```bash
# VTC on DINOv3 数据与权重下载
poetry run cofai-download examples/vtc/dinov3-vtc.manifest.txt
# VTC on DINOv2-reg4 数据与权重下载
poetry run cofai-download examples/vtc/dinov2-vtc.manifest.txt
```

数据集以压缩包形式下载到 `data/`，目前需**手动解压**到对应位置：
```bash
# VTC on DINOv3
unzip data/ADE20K.zip -d data/
unzip data/NYU_subset_for_training_depth_head.zip -d data/

# VTC on DINOv2-reg4
unzip data/ImageNet_val_sel2k.zip -d data/
unzip data/ADE20K.zip -d data/
```

解压后的最终目录结构如下：

```
CoFAI/
│
├─ data/                                  # 论文实验所用各数据子集
│   ├─ ADEChallengeData2016/
│   │   ├─ images/
│   │   └─ annotations/
│   ├─ NYU/
│   │   ├─ train/
│   │   └─ test/
│   │   ├─ nyu_train.txt/
│   │   └─ nyu_test.txt/
│
├─ weights/                               # 预训练权重
│   ├─ dinov3/
│   └─ VTC/
```

## Plan 测试方法

> Plan 文件使用 `${PROJECT_ROOT}` 引用权重/数据路径，运行前需先设置环境变量 `PROJECT_ROOT`（见项目根目录 README）。

每个 plan 都是自包含的（数据集、模型、任务头、码率点都写在文件里），用 `cofai-eval` 指定目录与文件名即可运行。

```bash
# ===== DINOv3-Large (slot-1) =====
poetry run cofai-eval --config-dir examples/vtc/plan --config-name ade20k-val--dinov3-large-slot-1--VTC-vbr args.multi_run=true    # ADE20K 分割, VTC VBR (qp 扫描)
poetry run cofai-eval --config-dir examples/vtc/plan --config-name ade20k-val--dinov3-large-slot-1--Bypass     # ADE20K 分割, Bypass 锚点 (无压缩)
poetry run cofai-eval --config-dir examples/vtc/plan --config-name nyuv2-val--dinov3-large-slot-1--VTC-vbr args.multi_run=true     # NYUv2 深度, VTC VBR (qp 扫描)
poetry run cofai-eval --config-dir examples/vtc/plan --config-name nyuv2-val--dinov3-large-slot-1--Bypass      # NYUv2 深度, Bypass 锚点 (无压缩)

# ===== DINOv2-Base-reg4 (slot-3) =====
poetry run cofai-eval --config-dir examples/vtc/plan --config-name ade20k-val--dinov2-base-reg4-slot-3--VTC-vbr args.multi_run=true            # ADE20K 分割, VTC VBR (qp 扫描)
poetry run cofai-eval --config-dir examples/vtc/plan --config-name ade20k-val--dinov2-base-reg4-slot-3--Bypass            # ADE20K 分割, Bypass 锚点 (无压缩)
poetry run cofai-eval --config-dir examples/vtc/plan --config-name imagenet-sel2k--dinov2-base-reg4-slot-3--VTC-vbr args.multi_run=true       # ImageNet sel2k 分类, VTC VBR (qp 扫描)
poetry run cofai-eval --config-dir examples/vtc/plan --config-name imagenet-sel2k--dinov2-base-reg4-slot-3--Bypass--rae-512px  # ImageNet sel2k 重建(RAE), Bypass 锚点 (无压缩)
```

参数说明：

- `--config-dir`: plan 文件所在目录（此处为 `examples/vtc/plan`）
- `--config-name`: plan 文件名（不含 `.yaml` 后缀）
- `args.multi_run=true`: 启用 plan 内 `multi_run` 定义的 qp 码率点扫描（默认 `false`，只跑单点）。VTC 的 plan 需加此项才会输出多码率结果；Bypass 的 plan 无 `multi_run` 段，不需要加
- 其余评估设置（数据集、模型、任务头）均写在 plan 文件内，无需命令行传参
- Bypass 的 plan 为锚点（`real: false`，码率≈0），衡量未压缩特征的任务上限


## 原测试方法（VTC on DINOv3）

> 配置文件使用 `${PROJECT_ROOT}` 引用权重/数据路径，运行前需先设置环境变量 `PROJECT_ROOT`（见项目根目录 README）。

```bash
# VTC DINOv3-Large VBR 测试 ADE20K 分割任务
poetry run python examples/vtc/run_eval.py \
    --config examples/vtc/config/dinov3_presets.yaml examples/vtc/config/eval_VTC-dionv3-large-vbr.yaml \
    --checkpoint "" \
    --task ade20k_val_seg \
    --head ade20k_seg_large_last1 \
    --cuda --recon 0  \
    --output_dir logs/ade20k_val_seg_MPC3-v3-large-vbr  --real

# VTC DINOv3-Large VBR 测试 NYUv2 val 深度估计任务
poetry run python examples/vtc/run_eval.py \
    --config examples/vtc/config/dinov3_presets.yaml examples/vtc/config/eval_VTC-dionv3-large-vbr.yaml \
    --checkpoint "" \
    --task nyuv2_val_dep \
    --head nyuv2_dep_large_last1 \
    --cuda --recon 0  \
    --output_dir logs/nyuv2_val_dep_MPC3-v3-large-vbr  --real

# VTC DINOv3-Large Bypass 测试 NYUv2 val 深度估计任务
poetry run python examples/vtc/run_eval.py \
    --config examples/vtc/config/dinov3_presets.yaml examples/vtc/config/eval_Bypass-dinov3-large.yaml \
    --checkpoint "" \
    --task nyuv2_val_dep \
    --head nyuv2_dep_large_last1 \
    --cuda --recon 0  \
    --output_dir logs/nyuv2_val_dep_Bypass-large-last1
```

参数说明：

- `--config`: 配置文件路径，可多个叠加
- `--task`（或 `--preset`）: 预定义的评估任务名称，需与配置中的 `eval_presets` 键一致（如 `ade20k_val_seg` / `nyuv2_val_dep`）
- `--head`: 头部模型名称，需要是预定义的头部模型
- `--quality`: 质量因子，仅用作任务标签
- `--cuda`: 使用CUDA
- `--recon`: 对于VTC模型，使用第几层分支的重建图像，当前可选[0,1,2]（历史参数，已弃用）
- `--real`: 启用真实熵编码，写入码流；否则使用码率估计，不写入码流
