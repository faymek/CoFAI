# CoFAI Documentation

CoFAI 是面向 AI 感知、理解与生成任务的多表征编码参考框架与 PyTorch 评测平台。
文档按“框架概念、当前实现、评测协议、API”分层组织，避免把长期概念、当前代码和
具体方法复现混在一起。

## 从这里开始

| 文档 | 回答的问题 |
|---|---|
| [CoFAI 框架概念](framework.md) | CoFAI 编码哪些表征，如何描述层、DU、AU、上下文和方案层级？ |
| [参考软件实现](reference_software.md) | 当前代码怎样实现 backbone、codec、head、slot、多流 DU 和 Token Grouping？ |
| [Engine 架构与数据流](engine.md) | 如何用 plan 构建并执行一次可复现评测？ |

具体方法的数据、权重、训练和运行命令位于仓库的 `examples/<method>/`。根目录
`README.md` 提供安装、Quick Start 和当前支持能力总览。

## Library API

- [cofai.models](api/models.md)
- [cofai.backbone](api/backbone.md)
- [cofai.latent_codecs](api/latent_codecs.md)
- [cofai.index_codecs](api/index_codecs.md)
- [cofai.token_grouping](api/token_grouping.md)
- [cofai.transforms](api/transforms.md)
- [cofai.engine](api/engine.md)
- [cofai.heads](api/heads.md)
- [cofai.layers](api/layers.md)
- [cofai.datasets](api/datasets.md)
- [cofai.losses](api/losses.md)
- [cofai.metrics](api/metrics.md)
- [cofai.utils](api/utils/utils.md)
- [数据与权重下载工具](api/utils/download.md)
