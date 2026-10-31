# ProTRAC

M2474：使用发布的源统计，在 ADE20K-C 语义分割和 AG-ReID-C 跨视角重识别上进行在线特征校准。

## 环境

按仓库根目录 [README.md](../../README.md) 安装 Poetry 环境并生成 `.env`。
以下命令从源码根目录执行，数据、权重和日志使用 `.env` 中的 `PROJECT_ROOT`。

## 下载数据与权重

下载清单包含 ADE20K、AG-ReID、强度 5 的扰动数据 ZIP、DINOv3 backbone、分割头及 ProTRAC 源统计，并校验 SHA256：

```bash
source .env
poetry run cofai-download examples/protrac/protrac.manifest.txt --root "$PROJECT_ROOT"
unzip "$PROJECT_ROOT/data/ADE20K.zip" -d "$PROJECT_ROOT/data/"
unzip "$PROJECT_ROOT/data/AG-ReID.zip" -d "$PROJECT_ROOT/data/"
unzip "$PROJECT_ROOT/data/ADE20K-C.zip" -d "$PROJECT_ROOT/data/"
unzip "$PROJECT_ROOT/data/AG-ReID-C.zip" -d "$PROJECT_ROOT/data/"
```

解压后的目录：

```text
${PROJECT_ROOT}/
├── data/
│   ├── ADEChallengeData2016/
│   │   ├── images/validation/
│   │   └── annotations/validation/
│   ├── ADE20K_val-c/<corruption>/5/validation/
│   └── AG-ReID/
│       ├── query_all_c0/
│       ├── query_all_c3/
│       ├── bounding_box_test_all_c0/
│       ├── bounding_box_test_all_c3/
│       └── query_corruption/<corruption>/5/query_all_c{0,3}/
└── weights/
    ├── dinov3/
    │   ├── backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors
    │   └── semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth
    └── protrac/
        ├── ade20k/{global_stats,class_stats,region_stats}.pt
        └── agreid/{vdt.pth,stats.pt}
```

## 评测

```bash
poetry run python examples/protrac/run_eval_semseg.py
poetry run python examples/protrac/run_eval_reid.py
```

默认评测强度 5：分割覆盖 15 个扰动域，每域 2,000 张图像；重识别覆盖 6 个扰动域和 AG、GA 两种协议，图库使用原图。分割在第 23 个 Transformer 块后校准，逐图预测后更新；重识别在第 11 个块后校准，每批 64 张查询完成检索后更新。每个域、每个协议开始时重置在线统计，模型使用 Bypass codec。

单域运行：

```bash
poetry run python examples/protrac/run_eval_semseg.py --corruptions gaussian_noise
poetry run python examples/protrac/run_eval_reid.py --corruptions motion_blur
```

结果写入 `logs/protrac/{semseg,reid}/result.json`，完整配置和环境版本写入同目录的 `config.yaml`。mIoU、mAP 和 Rank-1 均以百分数记录。

## 从原图生成扰动数据

也可以使用原图生成扰动数据：

```bash
poetry run python examples/protrac/generate_ade20k_c.py
poetry run python examples/protrac/generate_agreid_c.py
```

生成器固定种子为 `20260313`，默认强度为 5，跳过已有文件。分割保存为 JPEG 质量 75，重识别为质量 95。可用 `--corruptions` 选择域、`--workers` 设置生成进程数。
