# Qwen3VL CTC

## 基准与跨平台一致性（共识）

Qwen3VL CTC 的评测结果与 **Python 依赖版本**（如 `torch`、`transformers`）以及 **GPU 型号**（如 RTX 4090 / A6000）均有关联。跨单位 cross-check 时可能观察到数值或准确率上的差异。现阶段约定如下：

1. **统一基准数据**：各参与单位采用**同一组经 cross-check 确认的基准结果**作为对照；若不同 GPU 上存在可接受的计算误差，各单位在 CTC 口径上保持一致即可。
2. **跨 GPU 误差范围**：cross-check 中若出现跨 GPU 平台不一致，应事先**约定可接受的误差容限**（具体数值待各方讨论后填入本文档）。
3. **CE 阶段工程化**：定点化或其他工程实现可在 **CE（Conformance Environment）阶段**完成；IEEE 1857.11 **M441** 提案即在 CE 阶段完成相关约束。CTC 阶段可先以浮点参考实现跑通口径，CE 阶段的工程细节另行共识。

## 1. 通用测试条件

根据 AI M2462 的 CTC 讨论，Qwen3VL 通用测试条件选定如下：

| 项 | 取值 |
|----|------|
| 模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| 分割点 | **slot 9**（第一个 deepstack index，Layer 8 输出） |
| 输入尺度 | processor 动态 resize，`min_pixels=768×28×28`，`max_pixels=1536×28×28` |
| 下游任务 | 多选题 VQA / **MMStar** |
| 生成设置 | `do_sample=false`（greedy），`max_new_tokens=512` |
| 权重精度 | `bfloat16` |

> 注意：与 DINOv3 CTC 不同，Qwen3VL 是 Vision-Language Model；压缩对象是 vision encoder 在 slot 处的特征，解压后需继续跑完 vision encoder 并接入语言模型生成文本。

## 2. 下载数据与权重

### 数据集

MMStar 标注 TSV 通过下载清单 `qwen3vl-ctc.manifest.txt` 一键拉取并按 sha256 校验（默认从 `https://medialab.sjtu.edu.cn/files/CoFAI-share/` 按相对路径下载）。在仓库根目录执行：

```bash
poetry run cofai-download examples/ctc/qwen3vl-ctc.manifest.txt
```

下载后的目录结构：

```
CoFAI/
└─ data/
   └─ MMStar.tsv
```


### 模型权重

`Qwen3-VL-8B-Instruct` 由 HuggingFace `transformers` 在首次运行时自动拉取（约 16 GB）。若需离线使用，请提前 `huggingface-cli download Qwen/Qwen3-VL-8B-Instruct` 并在 plan 中设置 `local_files_only: true`。

> 运行前请设置 `PROJECT_ROOT`（项目根目录 `.env` 或 `install.py`），plan 通过 `${PROJECT_ROOT}` 引用数据路径。

## 3. 运行测试

通过统一评测引擎 `cofai-eval` 执行 plan，结果写入 `${PROJECT_ROOT}/logs/<plan-name>/result.json` 与 `config.yaml`。

```bash
# MMStar Bypass 基线（全量，约 11 分钟，峰值显存 ~19 GB）
CUDA_VISIBLE_DEVICES=0 poetry run cofai-eval \
  conf/plan/mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa.yaml
```

需要统计 **latent codec 复杂度**（参数量、编码/解码 FLOPs、codec 耗时）时，加上 `args.profile=true`；结果写入 `result.json` 末尾的 `codec_*` 字段。

参考结果（Bypass 不压缩，作为 VQA 精度上界）：

| Plan | bpp | 指标 |
|------|-----|------|
| `mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa` | 0.0 | accuracy **67.27%**（overall） |

## 4. 统一架构总览

与 DINOv3 CTC 对齐，Qwen3VL CTC 收敛为 **backbone + codec** 两层可插拔组件，由 **plan YAML** 组合：

| 层 | 角色 | 位置 |
|----|------|------|
| **Qwen3vlFeatureCodecModel** | 编排 backbone→codec→生成的流程，处理 VQA 任务分发与码率统计 | `cofai/models/base.py` |
| **Qwen3VLBackbone** | 视觉特征提取 / 重注入 + 语言模型生成，定义切分点（slot） | `cofai/backbone/qwen3vl.py` |
| **LatentCodec** | 对 breakpoint token 张量 `(B, L, C)` 做压缩/解压（与 DINO CTC 相同契约） | `cofai/latent_codecs/` |

与 DINOv3 CTC 的主要差异：

- 暂未抽取出独立 **head**：VQA 答案由 backbone 的 `decode_text` 直接输出文本。
- `MMStarDataset` 与其他图像任务统一输出 **`img`**；collate 后为 `(B,C,H,W)` 张量。文本 prompt 位于 `sample["vqa"]["prompt"]`，由 Qwen wrapper 读取。
- backbone `extract_features` 直接输出 **`(1, L, C)`** token，与 `DinoFeatureCodecModel` 一致；`token_res=(H,W)` 来自 `image_grid_thw`。

## 5. 模型组织：`Qwen3vlFeatureCodecModel`

定义于 `cofai/models/base.py`，与 `DinoFeatureCodecModel` 相似，由两个可插拔组件构成：

- **backbone**：`Qwen3VLBackbone`，`encode_image`（prepare + extract）→ `decode_text`。
- **codec**：当前 CTC 基线为 `BypassLatentCodec`（不压缩）。后续提案可替换为任意遵循第 6 节接口的 LatentCodec。

`forward_test` 流程：

1. `encode_image(image)` → `h_tokens` `(1, L, C)` 与 `token_res` `(H, W)`
2. `qwen_codec` 压缩/解压（Bypass 时直通）
3. `decode_text(h_hat, token_res=..., prompt=...)` → 生成文本
4. 返回 `task_feats["vqa"]` 供 `MMStarAccuracyMetric` 计分

## 6. LatentCodec 接口契约

与 DINO CTC **完全相同**：所有 codec 对编码器 token 张量 **`(B, L, C)`** 操作（`token_res` 供需要空间布局的 codec 自行 reshape，`qp` 供可变码率 codec 使用）：

```python
forward(h, token_res, qp)    -> {"h_hat", "likelihoods"}      # 训练 / 在线
compress(h, token_res, qp)   -> {"strings", "pstate"}         # 仅编码
decompress(strings, pstate)  -> {"h_hat"}                     # 仅解码
```

> `token_res` 由 `image_grid_thw` 在 model 层计算，随 `pstate` 在 compress/decompress 间传递；codec 若需空间布局应自行用 `token_res` reshape。


## 7. Backbone 与分割点（slot）

`Qwen3VLBackbone`（`cofai/backbone/qwen3vl.py`）遵循与 timm DINO backbone 一致的 slot 约定：

- `encode` = `visual.blocks[:slot]`
- `decode` = `visual.blocks[slot:]` + deepstack 特征收集 + 语言模型生成

slot 映射：

| 含义 | slot |
|------|------|
| 第一个 deepstack index（Layer 8 输出，**CTC 默认/上界**） | `slot9` |
| 更早的 breakpoint | `slot0` … `slot8` |

> `slot > 9` 暂时非法：decode 阶段无法恢复第一个 deepstack 特征。

## 8. 下游任务与指标

| 任务 | 数据集 | 指标 | 说明 |
|------|--------|------|------|
| 多选题 VQA | MMStar | `MMStarAccuracyMetric` | 匹配末位字母或 `A. <选项文本>` 格式；输出 overall + per-category 准确率 |

数据集与指标实现：

- `cofai/datasets/mmstar.py` — `MMStarDataset`
- `cofai/metrics/video_question_answer.py` — `MMStarAccuracyMetric`

plan 中通过 `task_configs` 声明（**不要用旧字段 `metric:`**）：

```yaml
task_configs:
  - label: vqa
    kind: vqa
    meter:
      type: MMStarAccuracyMetric
```

## 9. Plan 组合与命名

当前 CTC 基线 plan 位于 `conf/plan/`：

```
conf/plan/mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa.yaml
```

命名约定（与 DINO 四段式对齐）：

```
<dataset>__<backbone-feature>__<codec>__<task>.yaml
```

字段含义：

- `dataset`：benchmark 名称（`mmstar`）。
- `backbone-feature`：`<model-id>-slot<NN>`（如 `qwen3vl-8b-ins-slot9`）。
- `codec`：`Bypass` / `VTC` / `MPC` / …，可变码率可加 `-vbr`。
- `task`：下游任务（`vqa`）。

各提案的 plan 可放在 `examples/ctc/plan/`（或 `examples/<proposal>/plan/`），用 `cofai-eval` 直接指向对应 yaml。

plan 内部结构（节选）：

```yaml
# @package _global_
defaults:
  - /args: default          # 必须拉入，提供 args.cuda / args.device 等
  - /dataloader: default
  - _self_

name: mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa

dataset:
  type: MMStarDataset
  data_path: ${PROJECT_ROOT}/data/MMStar.tsv
  prompt_suffix: "Please select the correct answer ..."

task_configs:
  - label: vqa
    kind: vqa
    meter:
      type: MMStarAccuracyMetric

model:
  type: Qwen3vlFeatureCodecModel
  qwen_backbone:
    type: cofai.backbone.Qwen3VLBackbone
    model_path: Qwen/Qwen3-VL-8B-Instruct
    dtype: bfloat16
    min_pixels: 602112       # 768 * 28 * 28
    max_pixels: 1204224      # 1536 * 28 * 28
    max_new_tokens: 512
    do_sample: false
    slot: 9
  qwen_codec:
    type: cofai.latent_codecs.BypassLatentCodec

args:
  real: false
  output_dir: ${PROJECT_ROOT}/logs
```

## 10. 增加新组合的步骤

新增一种 Qwen3VL × codec 实验通常**无需写 Python**（除非引入全新 codec）：

1. 复制 `conf/plan/mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa.yaml` 为模板。
2. 替换 `model.qwen_backbone` 中的 `slot`、像素范围等 backbone 参数。
3. 将 `Bypass` 换为目标 LatentCodec 并填参数（需新增 codec 时，实现第 6 节接口并在 `cofai/latent_codecs/__init__.py` 导出）。
4. 按第 9 节命名，放入 `examples/ctc/plan/`。
5. 用 `cofai-eval` 跑测试。
