# GPS

本目录将 GPS 集成到 CoFAI，包含两部分评估：

1. AI M2405：基于 GPS 的多视角车辆检索评估。
2. AI M2460：Token Grouping 的结构码率、结构恢复和恢复复杂度评估。

GPS 的公共实现位于 `cofai/token_grouping/` 和
`cofai/token_codecs/token_selection_map.py`；数据集、模型和任务评估代码仅
服务于本示例，位于 `examples/gps/reid/`。

GPS 是论文 [Unsupervised Graph Partitioning Framework for Background Suppression in Multi-Query Vehicle Re-Identification](https://openaccess.thecvf.com/content/CVPR2026F/papers/Hu_Unsupervised_Graph_Partitioning_Framework_for_Background_Suppression_in_Multi-Query_Vehicle_CVPRF_2026_paper.pdf) 提出的一种 Token Grouping 技术。该论文目前已被 CVPR Findings 2026 接收，[正式的开源链接](https://github.com/HuYichun/GPS)。

## 环境准备

请先按照仓库根目录 `README.md` 完成 Poetry 环境安装。
所有命令均从 CoFAI 根目录执行，并使用 `poetry run python`。

## 数据与权重

从[网盘链接](https://medialab.sjtu.edu.cn/files/CoFAI-share/)中下载VeRI776、MuRi数据集，以及两个数据集对应的模型，组织为如下结构：

```text
CoFAI/
├── data/
│   ├── VeRi/
│   │   ├── image_train/
│   │   ├── image_query/
│   │   ├── image_test/
│   │   ├── keypoint_train.txt
│   │   └── keypoint_test.txt
│   ├── MuRI/
│   │   ├── train/
│   │   ├── query/
│   │   └── gallery/
└── weights/
    └── gps/
        ├── VeRi/transformer.pth
        ├── MuRI/transformer.pth
```

由于数据集的隐私问题，以下复现暂时不包括M2405中提及的 VERI-Wild 与 VERI-Wild 2.0 数据集

## 目录结构

```text
cofai/
├── token_grouping/
│   ├── gps.py                         # GPS 图划分 Token Grouping 公共实现
│   └── restore.py                     # 稀疏 token 与固定长度 token 恢复接口
└── token_codecs/
    └── token_selection_map.py         # Token selection map 编解码与一致性恢复

examples/gps/
├── config/
│   ├── VeRi/gps.yml                   # VeRi-776 数据、模型与评估配置
│   ├── MuRI/gps.yml                   # MuRI 数据、模型与评估配置
│   ├── token_grouping.yml             # M2460 码率与恢复实验配置
├── reid/
│   ├── backbone/
│   │   └── vit_pytorch.py             # GPS Transformer 主干与多视角 Token Grouping
│   ├── datasets/
│   │   ├── veri.py                    # VeRi-776 数据集读取
│   │   ├── muri.py                    # MuRI 数据集读取
│   │   ├── sampler_multiview.py       # 同 PID 三图 multi-query 采样
│   │   └── bases.py                   # ReID 数据集与图像样本基类
├── run_eval_reid.py                   # M2405 多视角车辆 ReID 评估入口
├── run_eval_token_grouping.py         # M2460 率性能与结构恢复评估入口
├── run_crosscheck.py                  # M2405 与 M2460 结果汇总入口
└── README.md                          # GPS 集成与复现说明
```



## AI M2405：ReID 评估

`run_eval_reid.py` 加载 GPS 主干和对应 checkpoint，执行 query 三视角融合、
gallery 单视角检索，并输出 mAP、Rank-1、Rank-5、Rank-10 和推理耗时。

VeRi-776：

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python examples/gps/run_eval_reid.py \
  --config examples/gps/config/VeRi/gps.yml \
  --output logs/gps/VeRi/result.json
```

MuRI：

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python examples/gps/run_eval_reid.py \
  --config examples/gps/config/MuRI/gps.yml \
  --output logs/gps/MuRI/result.json
```



## AI M2460：Token Grouping 结构评估

Token Grouping 公共入口位于 `run_eval_token_grouping.py`。配置文件为：

```text
examples/gps/config/token_grouping.yml
```

默认扫描 `rho=0.0,0.1,...,0.9`，记录以下字段：

- token 保留数量和保留率。
- 特征 payload bits、map bits、metadata bits 和 total BPFP。
- 结构侧信息占比和相对稠密码流的 rate saving。
- token map round-trip recovery。
- fixed-length restore 输出形状和恢复耗时。

VeRi-776：

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python examples/gps/run_eval_token_grouping.py \
  --config examples/gps/config/VeRi/gps.yml \
  --token-grouping-config examples/gps/config/token_grouping.yml \
  --output logs/gps/VeRi/token_grouping.csv
```

MuRI：

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python examples/gps/run_eval_token_grouping.py \
  --config examples/gps/config/MuRI/gps.yml \
  --token-grouping-config examples/gps/config/token_grouping.yml \
  --output logs/gps/MuRI/token_grouping.csv
```

该入口采用 FP16 定长特征作为码率代理，主要用于验证结构侧信息和恢复链路，不代表最终熵编码器的完整率失真结果。



## M2405 与 M2460 结果汇总

汇总当前 `logs/gps` 下已经生成的 JSON 和 CSV 并用于对比文档结果：

```bash
poetry run python examples/gps/run_crosscheck.py --root logs/gps
```