# MPC3 评估代码

这个评估系统提供了完整的MPC模型评估功能，一次编码，返回多个任务所需的特征，分别进行评估。

- 语义分割任务：mIoU
- 深度估计任务：rmse
- 压缩效率：BPP, 编码时间, 解码时间


## 下载数据与权重

如下是数据权重的分享链接

Share content: MPCompress-share
Link: https://pan.sjtu.edu.cn/web/share/2f9f14e05fa73c8742994aae67198dff
Extraction code: 1127

请下载链接中的数据与权重到对应文件夹，形成如下的目录结构。

```
MPCompress/
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
│       
│       
│       
│
├─ weights/                               # 预训练权重与下载脚本
│   ├─ dinov3/
│   │   ├─ dep_head/
│   │   ├─ seg_head/
│   │   
│   └─ MPC/
│       ├─ MPC3-v3-large.pth.tar
│       ├─ MPC3-v3-large-vbr.pth.tar
│       
```

## 测试方法

```bash
# MPC3 DINOv3-Lagre VBR 测试 ADE20K 分割任务
python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC3-v3-large-vbr.yaml \
    --checkpoint "" \
    --task ade20k_val_seg \
    --head ade20k_seg_large_last1 \
    --cuda --recon 0  \
    --output_dir eval_ade20k_val_seg_MPC3-v3-large-vbr  --real

# MPC3 DINOv3-Large VBR 测试 NYUv2 val 深度估计任务
python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC3-v3-large-vbr.yaml \
    --checkpoint "" \
    --task nyuv2_val_dep \
    --head nyuv2_dep_large_last1 \
    --cuda --recon 0  \
    --output_dir eval_nyuv2_val_dep_MPC3-v3-large-vbr  --real

# DINOv3-Large backbone 测试 NYUv2 val 深度估计任务
python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_Bypass-large-last1.yaml \
    --checkpoint "" \
    --task nyuv2_val_dep \
    --head nyuv2_dep_large_last1 \
    --cuda --recon 0  \
    --output_dir eval_nyuv2_val_dep_Bypass-large-last1
```

参数说明：

- `--config`: 配置文件路径，可多个叠加
- `--preset`: 预定义的评估任务名称，需要与配置文件中的任务名称一致
- `--head`: 头部模型名称，需要是预定义的头部模型
- `--quality`: 质量因子，仅用作任务标签
- `--cuda`: 使用CUDA
- `--recon`: 对于MPC模型，使用第几层分支的重建图像，当前可选[0,1,2]
- `--real`: 启用真实熵编码，写入码流；否则使用码率估计，不写入码流
