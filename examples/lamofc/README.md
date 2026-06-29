# LaMoFC

当前集成了论文 **“Feature Coding in the Era of Large Models” (LaMoFC)** 中的基线实验（基于 DINOv2 的特征压缩）。

## 环境配置

请参考 README.md 中的说明，先配置 Poetry 环境。

激活 Poetry 环境：
```sh
poetry shell
```

## 下载数据与权重

权重与数据集压缩包均通过下载清单 `dinov2-lamofc.manifest.txt` 一键拉取（默认从 `https://medialab.sjtu.edu.cn/files/CoFAI-share/` 按相对路径下载）。清单覆盖数据集 zip、DINOv2 预训练 backbone（small/giant）与分类/分割任务头。**`dino_timm_*` 配置经 timm/HuggingFace 在线加载 backbone，无需下载；仅 `dino_orig_slide_*` 配置需要本地 backbone。** LaMoFC 的编解码器为 VTM 标准编码器，需单独编译（见下文「准备 VTM」），不在清单内。在仓库根目录执行：

```bash
poetry run cofai-download examples/lamofc/dinov2-lamofc.manifest.txt
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
│   └─ std_codec/                         # VTM 标准编码器 (见「准备 VTM」)
```

## 准备 VTM

请下载 VTM 并解压到 `weights/std_codec/` 目录下，并且 checkout 到 VTM-21.0 版本。
```shell
cd weights/std_codec
git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
mv VVCSoftware_VTM VTM-21.0
cd VTM-21.0
git checkout VTM-21.0
```

接下来进行编译
```shell
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
```

## 运行示例

在项目根目录运行：

测试滑动窗口推理

```shell
CUDA_VISIBLE_DEVICES=0 python examples/lamofc/run_eval_slide.py \
    --config examples/lamofc/config/eval_base.yaml examples/lamofc/config/dino_orig_slide_patch_small_last1_vtm.yaml \
    --preset voc2012_sel20_seg \
    --head "voc2012_seg_small_last1" \
    --quality 1.0 \
    --cuda --real \
    --output_dir logs/voc2012_sel20_seg_dino_orig_slide_patch_small_last1_vtm
```

测试整图推理

```shell
CUDA_VISIBLE_DEVICES=0 python examples/lamofc/run_eval.py \
    --config examples/lamofc/config/eval_base.yaml examples/lamofc/config/dino_timm_patch_small_last1_vtm.yaml \
    --preset voc2012_sel20_seg \
    --head "voc2012_seg_small_last1" \
    --quality 1.0 \
    --cuda --real \
    --output_dir logs/voc2012_sel20_seg_dino_timm_patch_small_last1_vtm

```

更多的 preset 和 head 配置请参考 `examples/lamofc/config/eval_base.yaml`。
