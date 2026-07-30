# GPS

本目录将 GPS 集成到 CoFAI，包含两部分评估：

1. AI M2405：基于 GPS 的多视角车辆检索评估。
2. AI M2460：Token Grouping 的真实原生 dtype 码率和结构侧信息评估。

GPS 的公共组件位于 `cofai/backbone/gps_transreid.py`、
`cofai/heads/transreid_jpm.py`、`cofai/token_grouping/` 和
`cofai/index_codecs/`。数据集、checkpoint 适配和任务评估代码仅服务于
本示例，位于 `examples/gps/reid/`。

本示例使用与 DINO CTC 相同的特征切分思想，但不会修改已经形成共识的
`DinoFeatureCodecModel`。GPS 使用平行的探索性模型
`CommonFeatureCodecModel`：

```text
GPSTransReIDBackbone.encode（内部执行多层 GPS）
├─ compact features → RawDtypeCodec（当前 dtype=float16）
└─ selection indices → AdaptiveBitmapIndexCodec
→ post_process=None（预留变量，当前直接透传）
→ GPSTransReIDBackbone.decode（TransReID 全局与 JPM 局部分支）
→ TransReIDJPMHead（原 bottleneck 与 embedding 拼接）
```

GPS 必须在多个 Transformer 层内部读取 attention，因此它属于 backbone
encoder 的组成部分，不能被提取成一次性的串行预处理。当前 ReID decoder
直接消费 compact token sequence，不需要恢复成完整二维 token 网格。
`post_process` 仅保留为空变量；稠密恢复应等待相应算法和下游任务实现，
本集成不假设其形式或插入位置。

`CommonFeatureCodecModel` 不引入另一套公共接口。普通 backbone 的 `encode`
仍可只返回特征张量；GPS backbone 在 `h` 和 `pstate` 之外返回未编码的
`index_sets`，由模型配置中的 `index_codecs` 生成结构侧码流。backbone
不直接写 bytes。最终 `compress` 输出仍是文档规定的 CodedUnit：

```python
coded_unit = {"strings": ..., "pstate": ...}
```

ReID 的 `pid`、`camid`、图像路径和三视角分组关系属于数据 batch meta。
GPS encoder 内部生成的 `lsort` 只用于特征分组，不进入 Feature DU；
evaluator 从 collate 后的 batch meta 推导 grouped embedding 对应的样本。

特征码流使用真实 `RawDtypeCodec` 序列化，包括一个 CLS token 和全部 retained
patch tokens。该 codec 通过 PyTorch 原生浮点 dtype 完成数值量化，再直接传输
连续原始字节；当前配置为 `float16`，也支持 `bfloat16`、
`float8_e4m3fn` 和 `float8_e5m2`。码流保持 eval 引擎现有的扁平结构：

```python
strings = {
    "feature": [[feature_bytes]],
    "selection_map": [[map_bytes]],
}
```

总码率只由这两个真实 byte stream 相加得到。selection map 为兼容 GPS
结构码率定义而保留；当前 `post_process=None` 的 ReID 推理不会消费它。

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
├── backbone/
│   └── gps_transreid.py                # GPS 多视角 ViT 与 CoFAI backbone 边界
├── heads/
│   └── transreid_jpm.py                # TransReID JPM embedding head
├── models/
│   └── common.py                       # 探索性 CommonFeatureCodecModel
├── latent_codecs/
│   └── raw_dtype.py                    # 原生 dtype 数值量化与原始特征码流
├── token_grouping/
│   └── gps.py                         # GPS 图划分 Token Grouping 公共实现
└── index_codecs/
    └── adaptive_bitmap_index.py       # Bitmap/index-list 自适应索引编解码

examples/gps/
├── model.py                            # Common model + GPS backbone 构建
├── config/
│   ├── VeRi/gps.yml                   # VeRi-776 数据、模型与评估配置
│   ├── MuRI/gps.yml                   # MuRI 数据、模型与评估配置
│   ├── token_grouping.yml             # M2460 码率与 map 一致性配置
├── reid/
│   ├── datasets/
│   │   ├── veri.py                    # VeRi-776 数据集读取
│   │   ├── muri.py                    # MuRI 数据集读取
│   │   ├── sampler_multiview.py       # 同 PID 三图 multi-query 采样
│   │   └── bases.py                   # ReID 数据集与图像样本基类
├── run_eval_reid.py                   # M2405 多视角车辆 ReID 评估入口
├── run_eval_token_grouping.py         # M2460 率性能与 map 评估入口
├── run_crosscheck.py                  # M2405 与 M2460 结果汇总入口
└── README.md                          # GPS 集成与复现说明
```



## AI M2405：ReID 评估

`run_eval_reid.py` 加载 GPS 主干和对应 checkpoint，执行配置 dtype 的真实
compress/decompress、query 三视角融合和 gallery 单视角检索，并输出 mAP、
Rank-1、Rank-5、Rank-10、各 stream bits 和推理耗时。

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
- 完整 compact tensor（CLS + retained patches）的 feature bits、map bits 和
  total BPFP。
- 结构侧信息占比和相对稠密码流的 rate saving。
- 实际传输 token map 码流的 round-trip 与解码耗时。它只校验结构侧信息，
  不恢复 patch tokens，也不进入 ReID 质量推理路径。

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

该入口实际执行原生 dtype 序列化和反序列化；汇总公式会从真实 `feature`
码流自动推导每个数值的 bit depth，并与 `selection_map` 字节数逐项交叉
检查。当前 `dtype=float16` 是本提案的直接传输工作点，而不是熵编码结果。



## M2405 与 M2460 结果汇总

汇总展示当前 `logs/gps` 下已经生成的 JSON 和 CSV：

```bash
poetry run python examples/gps/run_crosscheck.py --root logs/gps
```
