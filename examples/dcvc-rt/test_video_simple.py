# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import os
import time
import torch
from cofai.utils.common import (
    str2bool,
    generate_summary,
    dump_json,
    set_torch_env,
)
from CoFAI.cofai.entropy_models.dcvc_base import DmcCompressionModel
from CoFAI.cofai.models.dcvcrt.pframe import DMCP
from CoFAI.cofai.models.dcvcrt.iframe import DMCI
from CoFAI.cofai.models.dcvcrt.video import DCVC_RT_Video
from CoFAI.cofai.metrics.utils import DataFrameRecords

from CoFAI.cofai.datasets.video_reader import PngSequenceVideoReader, YUV420VideoReader
from CoFAI.cofai.datasets.video_writer import PngSequenceVideoWriter, YUV420VideoWriter
from cofai.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from CoFAI.cofai.utils.transforms import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_444_to_420,
    ycbcr420_to_444_np,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")
    parser.add_argument("--qp_i", type=int, nargs="+")
    parser.add_argument("--qp_p", type=int, nargs="+")
    parser.add_argument("--test_config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out_bin")
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()
    return args


def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_video_reader(args):
    if args["src_type"] == "png":
        src_reader = PngSequenceVideoReader(
            args["src_path"], args["src_width"], args["src_height"]
        )
    elif args["src_type"] == "yuv420":
        src_reader = YUV420VideoReader(
            args["src_path"], args["src_width"], args["src_height"]
        )
    return src_reader


def get_video_writer(args):
    if args["src_type"] == "png":
        recon_writer = PngSequenceVideoWriter(
            args["out_dir"], args["src_width"], args["src_height"]
        )
    elif args["src_type"] == "yuv420":
        # output_yuv_path = args["out_rec_path"].replace(".yuv", "_xxxkbps.yuv")
        recon_writer = YUV420VideoWriter(
            args["out_rec_path"], args["src_width"], args["src_height"]
        )
    return recon_writer


def calc_distortion(args, x_hat, frame):
    if args["src_type"] == "yuv420":
        y_rec, uv_rec = yuv_444_to_420(x_hat)
        y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        y_rec = y_rec[0, :, :]
        u_rec = uv_rec[0, :, :]
        v_rec = uv_rec[1, :, :]
        psnr_y = calc_psnr(frame.y, y_rec)
        psnr_u = calc_psnr(frame.u, u_rec)
        psnr_v = calc_psnr(frame.v, v_rec)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
        if args["calc_ssim"]:
            msssim_y = calc_msssim(frame.y, y_rec)
            msssim_u = calc_msssim(frame.u, u_rec)
            msssim_v = calc_msssim(frame.v, v_rec)
            msssim = (6 * msssim_y + msssim_u + msssim_v) / 8
        else:
            msssim, msssim_y, msssim_u, msssim_v = 0.0, 0.0, 0.0, 0.0

        result = {
            "psnr": psnr,
            "psnr_y": psnr_y,
            "psnr_u": psnr_u,
            "psnr_v": psnr_v,
            "msssim": msssim,
            "msssim_y": msssim_y,
            "msssim_u": msssim_u,
            "msssim_v": msssim_v,
        }
        return result
    else:
        assert args["src_type"] == "png"
        rgb_rec = ycbcr2rgb(x_hat)
        rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        psnr = calc_psnr(frame.rgb, rgb_rec)
        if args["calc_ssim"]:
            msssim = calc_msssim_rgb(frame.rgb, rgb_rec)
        else:
            msssim = 0.0
        result = {"psnr": psnr, "msssim": msssim}
        return result


def run_one_point_with_stream(video_model: DCVC_RT_Video, args: dict):
    video_reader = get_video_reader(args)

    compress_args = {
        "src_type": args["src_type"],
        "src_height": args["src_height"],
        "src_width": args["src_width"],
        "frame_num": args["frame_num"],
        "qp_i": args["qp_i"],
        "qp_p": args["qp_p"],
        "reset_interval": args["reset_interval"],
        "intra_period": args["intra_period"],
        "out_bin_path": args["out_bin_path"],
    }

    decompress_args = {
        "src_height": args["src_height"],
        "src_width": args["src_width"],
        "frame_num": args["frame_num"],
        "out_bin_path": args["out_bin_path"],
    }

    # per-frame encoding
    df_records = DataFrameRecords()
    sequence_start_time = time.time()
    video_model.init_compress(compress_args)
    with torch.no_grad():
        for frame_idx in range(args["frame_num"]):
            _frame = video_reader.read_one_frame()
            x = np_image_to_tensor(_frame.yuv444, video_model.device)
            x = x.to(torch.float16)
            frame_start_time = time.time()
            encoded = video_model.compress_frame_then_write(x)
            frame_time = time.time() - frame_start_time

            frame_result = {
                "_id": frame_idx,
                "frame_type": encoded["pstate"]["frame_type"],
                # "bits": frame_enc_info["bits"],
                "bpp": encoded["bits"]
                / (args["src_height"] * args["src_width"]),
                "encoding_time": frame_time,
            }
            df_records.update(frame_result)

    total_bytes = video_model.close_write()

    # per-frame decoding
    video_reader = get_video_reader(args)
    if args["save_decoded_frame"]:
        video_writer = get_video_writer(args)

    sequence_start_time = time.time()
    video_model.init_decompress(decompress_args)
    with torch.no_grad():
        for i in range(args["frame_num"]):
            frame_start_time = time.time()
            decoded = video_model.decompress_frame_by_read()
            frame_time = time.time() - frame_start_time
            x_hat = decoded["x_hat"]
            _frame = video_reader.read_one_frame()
            distortions: dict = calc_distortion(args, x_hat, _frame)

            frame_result = {
                "_id": i,
                "decoding_time": frame_time,
                **distortions,
            }
            df_records.update(frame_result)

            if args["save_decoded_frame"]:
                video_writer.write_one_frame_from_tensor(x_hat, "yuv444")

    video_model.close_read()
    if args["save_decoded_frame"]:
        video_writer.close()

    summary: dict = generate_summary(df_records.df)
    summary.update(
        test_time=time.time() - sequence_start_time,
        frame_pixel_num=args["src_height"] * args["src_width"],
    )

    with open(args["out_json_path"], "w") as fp:
        json.dump(summary, fp, indent=2)

    return summary


def main():
    args = parse_args()

    with open(args.test_config) as f:
        config = json.load(f)

    set_torch_env()

    video_model = DCVC_RT_Video(
        dmci_codec={
            "type": "cofai.models.dcvcrt.iframe.DMCI",
            "load_path": "/home/faymek/DCVC/checkpoints/cvpr2025_image.pth.tar",
        },
        dmcp_codec={
            "type": "cofai.models.dcvcrt.pframe.DMCP",
            "load_path": "/home/faymek/DCVC/checkpoints/cvpr2025_video.pth.tar",
        },
    )
    video_model = video_model.to("cuda")
    video_model.eval()
    video_model.update()
    video_model.half()

    results = []

    qp_i = args.qp_i
    qp_p = args.qp_p if args.qp_p is not None else qp_i
    assert len(qp_i) == len(qp_p)

    root_path = config["root_path"]
    config = config["test_classes"]

    for ds_name in config:
        if config[ds_name]["test"] == 0:
            continue
        for seq in config[ds_name]["sequences"]:
            for rate_idx in range(len(qp_i)):
                src_path = os.path.join(root_path, config[ds_name]["base_path"], seq)

                codec_args = {
                    "qp_i": qp_i[rate_idx],
                    "qp_p": qp_p[rate_idx],
                    "reset_interval": 64,
                    "src_path": src_path,
                    "src_type": config[ds_name]["src_type"],
                    "src_height": config[ds_name]["sequences"][seq]["height"],
                    "src_width": config[ds_name]["sequences"][seq]["width"],
                    "intra_period": -1,
                    "frame_num": config[ds_name]["sequences"][seq]["frames"],
                    "calc_ssim": False,
                    "write_stream": True,
                    "save_decoded_frame": False,
                }

                out_dir = os.path.join(args.out_dir, ds_name)
                os.makedirs(out_dir, exist_ok=True)

                codec_args["out_dir"] = out_dir
                out_name = f"{seq}_q{qp_i[rate_idx]}"
                codec_args["out_bin_path"] = os.path.join(out_dir, out_name + ".bin")
                codec_args["out_rec_path"] = os.path.join(out_dir, out_name + ".yuv")
                codec_args["out_json_path"] = os.path.join(out_dir, out_name + ".json")

                result = run_one_point_with_stream(video_model, codec_args)

                result["ds_name"] = ds_name
                result["seq"] = seq
                result["rate_idx"] = rate_idx
                result["qp_i"] = codec_args["qp_i"]
                result["qp_p"] = codec_args["qp_p"]

                results.append(result)

    log_result = {}
    for res in results:
        # {"ds_name.seq_name.rate_idx": summary}
        ds_name = res["ds_name"]
        seq_name = res["seq"]
        rate_idx = res["rate_idx"]
        log_result[f"{ds_name}--{seq_name}--{rate_idx:03d}"] = res

    out_json_dir = os.path.dirname(args.json_path)
    if len(out_json_dir) > 0:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.json_path, "w") as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)


if __name__ == "__main__":
    main()


"""
example:

CUDA_VISIBLE_DEVICES=0  python examples/dcvc-rt/test_video_simple.py --qp_i 0 63  --test_config examples/dcvc-rt/dataset_config_test.json --json_path output.json

"""
