# VQFC

当前集成了论文 **“Transform-Free Feature Coding via Entropy-Constrained Vector Quantization” (VQFC)** 中的基线实验（基于 DINOv2 的特征压缩）。

## 下载数据与权重

权重与数据集压缩包均通过下载清单 `dinov2-vqfc.manifest.txt` 一键拉取（默认从 `https://medialab.sjtu.edu.cn/files/CoFAI-share/` 按相对路径下载）。清单覆盖 DINOv2 ViT-G/14 backbone、分类/分割任务头，以及 VQFC 的全部码本权重（cls: 8/16/32/512/2048，seg: 16/64/256/512 及 lmbda3 变体）。在仓库根目录执行：

```bash
poetry run cofai-download examples/vqfc/dinov2-vqfc.manifest.txt
```

数据集以压缩包形式下载到 `data/`，目前需**手动解压**到对应位置：

```bash
# ImageNet sel100 分类
unzip data/ImageNet_val_sel100.zip -d data/
# VOC2012 分割（含 VOC2012_sel20.txt 选图列表）
unzip data/VOC2012.zip -d data/
```

解压后的最终目录结构如下：

```
CoFAI/
│
├─ data/                                  # 评测所用数据子集
│   ├─ ImageNet_val_sel100/
│   └─ VOC2012/
│
├─ weights/                               # 预训练权重
│   ├─ dinov2/                            # backbone + 分类/分割任务头
│   └─ VQFC/                              # VQFC 码本 (.pth.tar)
```

## cofai-eval 评测

**cofai-eval** 是基于 Hydra 配置的统一评测入口，通过 **`poetry run cofai-eval`** 命令串联数据集、模型与指标（详见 [docs/engine.md](../../docs/engine.md)）。VQFC 对应 plan 位于 `conf/plan/`，可与下方脚本评测对照使用。

```bash
# VOC2012 sel20 语义分割
CUDA_VISIBLE_DEVICES=0 poetry run cofai-eval \
  conf/plan/voc2012-sel20--VQFC-dinov2-giant-seg.yaml \
  args.real=true args.multi_run=true

# ImageNet sel100 分类
CUDA_VISIBLE_DEVICES=0 poetry run cofai-eval \
  conf/plan/imagenet-sel100--VQFC-dinov2-giant-cls.yaml \
  args.real=true args.multi_run=true
```

> 注意：对于语义分割任务，这里的 plan 暂不支持 VQFC 的滑动窗口推理，而是使用 Resize 到固定尺寸的预处理，因此测下来结果会比较差，这里仅作为参考。为获得更好的结果，可以使用下边的 run_eval_slide 进行推理测试。

## 集成测试

请按照项目根目录 `README.md` 的说明完成环境配置：安装 Poetry、创建 `.env` 文件（含 `PROJECT_ROOT` 变量），并准备测试数据与权重。

其中 segmentation 任务使用了滑动窗口推理，classification 任务使用了整图推理。

**VQFC 推理示例**：

```bash 
CUDA_VISIBLE_DEVICES=0 python examples/vqfc/run_eval_slide.py \
    --config examples/vqfc/config/eval_base.yaml examples/vqfc/config/dino_orig_slide_giant_seg_vqfc_64.yaml \
    --preset voc2012_sel20_seg \
    --head voc2012_seg_giant_last1 \
    --quality 0 \
    --cuda --real \
    --output_dir logs/voc2012_sel20_seg_dino_orig_slide_giant_seg_vqfc_64

CUDA_VISIBLE_DEVICES=0 python examples/vqfc/run_eval_slide.py \
      --config examples/vqfc/config/eval_base.yaml examples/vqfc/config/dino_orig_slide_giant_cls_vqfc_512.yaml \
      --preset imagenet_sel100_cls \
      --head imagenet_cls_giant_last1 \
      --quality 0 \
      --cuda --real \
      --output_dir logs/imagenet_sel100_cls_dino_orig_slide_giant_cls_vqfc_512
```

