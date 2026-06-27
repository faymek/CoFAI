# MPC3 评估代码

这个评估系统提供了完整的MPC模型评估功能，一次编码，返回多个任务所需的特征，分别进行评估。

- 语义分割任务：mIoU
- 深度估计任务：rmse
- 压缩效率：BPP, 编码时间, 解码时间


## 下载数据与权重

权重与数据集压缩包均通过下载清单 `dinov3-vtc.manifest.txt` 一键拉取并按 sha256 校验（默认从 `https://medialab.sjtu.edu.cn/files/CoFAI-share/` 按相对路径下载）。在仓库根目录执行：

```bash
# 下载权重与数据集压缩包（缺失/损坏才下载），并校验
poetry run cofai-download examples/vtc/dinov3-vtc.manifest.txt
```

> 数据集以压缩包形式下载到 `data/`，目前需**手动解压**到对应位置：
>
> ```bash
> unzip data/ADE20K.zip -d data/
> unzip data/NYU_subset_for_training_depth_head.zip -d data/
> ```

解压后的最终目录结构如下：

```
CoFAI/
│
├─ data/                                  # 论文实验所用各数据子集
│   ├─ ADEChallengeData2016/
│   │   ├─ images/
│   │   └─ annotations/
│   ├─ NYU/
│       ├─ train/
│       └─ test/
│       ├─ nyu_train.txt/
│       └─ nyu_test.txt/
│
├─ weights/                               # 预训练权重
│   ├─ dinov3/
│   │   ├─ backbone/
│   │   │   └─ dinov3_vitl16_pretrain_lvd1689m.safetensors   # DINOv3-Large backbone
│   │   ├─ semseg_head/
│   │   │   └─ dinov3_vitl16_semseg_ade20k_linear_head.pth   # ADE20K 分割头
│   │   └─ dpt_head/
│   │       └─ dinov3_vitl16_depth_nyuv2_linear_head.pth     # NYUv2 深度头
│   └─ VTC/
│       ├─ dinov3-vitl16-slot-1--VTC.pth.tar                # MPC3 定码率
│       └─ dinov3-vitl16-slot-1--VTC-vbr.pth.tar            # MPC3 可变码率(VBR)
```

## 测试方法

> 配置文件使用 `${PROJECT_ROOT}` 引用权重/数据路径，运行前需先设置环境变量 `PROJECT_ROOT`（见项目根目录 README）。

```bash
# MPC3 DINOv3-Large VBR 测试 ADE20K 分割任务
poetry run python examples/vtc/run_eval.py \
    --config examples/vtc/config/dinov3_presets.yaml examples/vtc/config/eval_MPC3-v3-large-vbr.yaml \
    --checkpoint "" \
    --task ade20k_val_seg \
    --head ade20k_seg_large_last1 \
    --cuda --recon 0  \
    --output_dir logs/ade20k_val_seg_MPC3-v3-large-vbr  --real

# MPC3 DINOv3-Large VBR 测试 NYUv2 val 深度估计任务
poetry run python examples/vtc/run_eval.py \
    --config examples/vtc/config/dinov3_presets.yaml examples/vtc/config/eval_MPC3-v3-large-vbr.yaml \
    --checkpoint "" \
    --task nyuv2_val_dep \
    --head nyuv2_dep_large_last1 \
    --cuda --recon 0  \
    --output_dir logs/nyuv2_val_dep_MPC3-v3-large-vbr  --real

# DINOv3-Large backbone 测试 NYUv2 val 深度估计任务
poetry run python examples/vtc/run_eval.py \
    --config examples/vtc/config/dinov3_presets.yaml examples/vtc/config/eval_Bypass-large-last1.yaml \
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
- `--recon`: 对于MPC模型，使用第几层分支的重建图像，当前可选[0,1,2]
- `--real`: 启用真实熵编码，写入码流；否则使用码率估计，不写入码流
