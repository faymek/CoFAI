# MPC 评估代码

这个评估系统提供了完整的MPC模型评估功能，一次编码，返回多个任务所需的特征，分别进行评估。

- 图像重建任务：PSNR, MS-SSIM, LPIPS, CLIP-SIM, FID
- 图像分类任务：Top-1 Accuracy, Top-5 Accuracy
- 语义分割任务：mIoU
- 压缩效率：BPP, 编码时间, 解码时间


## 配置环境

按照项目 README.md 中的说明配置环境。

你可能需要设置 HuggingFace 的镜像地址，以更快地加载模型。
```
export HF_ENDPOINT=https://hf-mirror.com 
```

## 下载数据与权重

如下是数据权重的分享链接

Share content: CoFAI-share
Link: https://pan.sjtu.edu.cn/web/share/2f9f14e05fa73c8742994aae67198dff
Extraction code: 1127

请下载链接中的数据与权重到对应文件夹，形成如下的目录结构。

```
CoFAI/
│
├─ data/                                  # 论文实验所用各数据子集
│   ├─ ADEChallengeData2016/
│   │   ├─ images/
│   │   └─ annotations/
│   ├─ ImageNet_val_sel2k/
│   │   ├─ img/
│   │   └─ imagenet_val_labels.txt
│   └─ VOC2012/
│       ├─ Annotations/
│       ├─ .../
│       └─ JPEGImages/
│
├─ weights/                               # 预训练权重与下载脚本
│   ├─ dinov2/
│   │   ├─ clf_head/
│   │   ├─ seg_head/
│   │   └─ download_pretrained.sh
│   └─ MPC/
│       ├─ MPC2-v3-base-vbr-pruned.pth.tar
│       ├─ MPC2-v3-small-vbr-pruned.pth.tar
│       └─ MPC2-v3-large-vbr-pruned.pth.tar
```






## 测试方法（专用脚本 `run_eval.py`，参考）

在仓库根目录执行（需已设置 `PROJECT_ROOT`，见项目 README）。默认按 `--quality` **只跑一轮**；若要与 YAML 中的 `multi_run` 一致、扫多个 quality，请加 **`--multi-run`**。

```bash
# MPC2 DINO Large VBR 测试 ImageNet 分类任务
CUDA_VISIBLE_DEVICES=0 poetry run python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC2-v3-large-vbr.yaml \
    --preset imagenet_sel2k_cls \
    --head "imagenet_cls_large_last4" \
    --quality 0 \
    --cuda --recon 0 --real \
    --output_dir logs/imagenet_sel2k_cls_mpc2_dino_large_vbr

# MPC2 DINO Base VBR 测试 VOC2012 分割任务
CUDA_VISIBLE_DEVICES=0 poetry run python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC2-v3-base-vbr.yaml \
    --preset voc2012_val_seg \
    --head "voc2012_seg_base_last4" \
    --quality 16 \
    --cuda --recon 0 --real \
    --output_dir logs/voc2012_val_seg_mpc2_dino_base_vbr

# MPC2 DINO Small VBR 测试 ADE20K 分割任务
CUDA_VISIBLE_DEVICES=0 poetry run python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC2-v3-small-vbr.yaml \
    --preset ade20k_val_seg \
    --head "ade20k_seg_small_last4" \
    --quality 32 \
    --cuda --recon 0 --real \
    --output_dir logs/ade20k_val_seg_mpc2_dino_small_vbr

# MPC2 DINO Base reg4 VBR 测试 RAE 重建图像
CUDA_VISIBLE_DEVICES=0 poetry run python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC2-v3-base-vbr-reg4.yaml \
    --preset imagenet_sel2k_cls \
    --head rae_dinov2_base_reg4_512px \
    --recon 1 \
    --cuda \
    --multi-run \
    --output_dir logs/rae_mpc2_base_vbr_multirun_imagenet2k
```

参数说明：

- `--config`: 配置文件路径，可多个叠加
- `--preset`: 预定义的评估任务名称，需要与配置文件中的任务名称一致
- `--head`: 头部模型名称，需要是预定义的头部模型
- `--quality`: 码率/量化档位索引（整数，传给模型的 `qp`；脚本会将 `"1.0"` 规范为 `1`）
- `--multi-run`: 若合并后的配置含 `multi_run`，对每个档位各跑一次并写出汇总；**不加则只跑当前 `--quality` 一次**
- `--cuda`: 使用CUDA
- `--recon`: 对于MPC模型，使用第几层分支的重建图像，当前可选[0,1,2]
- `--real`: 启用真实熵编码，写入码流；否则使用码率估计，不写入码流


## 通用评测（cofai-eval plan）

推荐使用统一评测引擎 **`poetry run cofai-eval`**（与 `python -m cofai.engine.run_eval` 相同）。在仓库根目录执行，Plan 位于 **`conf/plan/`**，通用参数与语义见 **[docs/engine.md](../../docs/engine.md)**。

```bash
# MPC2-v3 base VBR reg4 + RAE，ImageNet sel2k（plan 内 multi_run 扫 qp 0/8/…/64）
CUDA_VISIBLE_DEVICES=0 poetry run cofai-eval \
  conf/plan/imagenet-sel2k--MPC2-v3-base-vbr-reg4-rae-512px.yaml \
  args.cuda=true \
  args.multi_run=true
```

常用覆盖（Hydra 语法）：

- `args.cuda=true` / `args.real=true`（是否真实熵编解码，默认 `real: false`）
- `args.multi_run=true`（按 plan 的 `multi_run` 多档 quality，结果在 `logs/<plan.name>/q<qp>/` 与 `summary.json`）
- `args.quality=16`（单档，且不加 `args.multi_run=true` 时）
- `args.output_dir=logs/my_run`（可选；默认 `logs/<plan.name>/`）
- `args.max_samples=10`（调试）

---