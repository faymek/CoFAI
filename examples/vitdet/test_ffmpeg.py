# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

import math
import torch
import torch.nn.functional as F
import types
import sys
import json
from dotenv import load_dotenv
from CoFAI.cofai.metrics.utils import DictAverageMeter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from CoFAI.examples.vitdet.ffmpeg_token_codec import FFmpegVitTokenCodec

load_dotenv()

project_root = os.environ.get("PROJECT_ROOT")
layer_idx = int(os.environ.get("COMPRESS_LAYER", 6))
codec = str(os.environ.get("CODEC", "libx264"))
qp = int(os.environ.get("QP", 12))
preset = str(os.environ.get("PRESET", "placebo"))

ffmpeg_path = "ffmpeg"
work_dir = f"{project_root}/logs/ViTDet/vit-b_layer{layer_idx}/{codec}_{qp}_{preset}"
log_path = f"{project_root}/logs/ViTDet/vit-b_layer{layer_idx}/ffmpeg_stats.txt"
codec_stats = DictAverageMeter()


token_codec = FFmpegVitTokenCodec(
    ffmpeg_path=ffmpeg_path, work_dir=work_dir, codec=codec, qp=qp, preset=preset
)


def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def vit_forward(self, x):
    global token_codec
    global layer_idx
    global codec_stats
    iB, iC, iH, iW = x.shape

    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )

    for i, blk in enumerate(self.blocks, start=1):
        x = blk(x)

        if i == layer_idx:  # 第三层
            # [B, H, W, C] -> [B, C, H, W]
            _, fH, fW, fC = x.shape
            feat = x.permute(0, 3, 1, 2)
            out_dec = token_codec(feat)
            rec_feat = out_dec["x_hat"]
            x = rec_feat.permute(0, 2, 3, 1)

            # codec stats
            num_bytes = out_dec["num_bytes"]
            feat_numels = feat.numel()
            mse = torch.mean((feat - rec_feat) ** 2).item()
            record = {
                "mse": mse,
                "bpp": num_bytes * 8 / (iB * iH * iW),
                "bpfp": num_bytes * 8 / feat_numels,
                "num_bytes": num_bytes,
                "feat_numels": feat_numels,
                "resolution": f"{iH}x{iW} -> {fC}x{fH}x{fW}",
            }
            codec_stats.update(record)
            print(record)

    x = x.permute(0, 3, 1, 2)
    return x


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show-dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        if "tta_model" not in cfg:
            warnings.warn(
                "Cannot find ``tta_model`` in config, we will set it as default."
            )
            cfg.tta_model = dict(
                type="DetTTAModel",
                tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
            )
        if "tta_pipeline" not in cfg:
            warnings.warn(
                "Cannot find ``tta_pipeline`` in config, we will set it as default."
            )
            test_data_cfg = cfg.test_dataloader.dataset
            while "dataset" in test_data_cfg:
                test_data_cfg = test_data_cfg["dataset"]
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(type="RandomFlip", prob=1.0),
                        dict(type="RandomFlip", prob=0.0),
                    ],
                    [
                        dict(
                            type="PackDetInputs",
                            meta_keys=(
                                "img_id",
                                "img_path",
                                "ori_shape",
                                "img_shape",
                                "scale_factor",
                                "flip",
                                "flip_direction",
                            ),
                        )
                    ],
                ],
            )
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith((".pkl", ".pickle")), (
            "The dump file must be a pkl file."
        )
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))

    # start testing
    runner.model.backbone.forward = types.MethodType(vit_forward, runner.model.backbone)
    metrics = runner.test()
    print(metrics)
    print(codec_stats.average())
    results = {
        "task": metrics,
        "codec": codec_stats.average(),
    }
    with open(log_path, "a") as f:
        json.dump(results, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    main()
