import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from CoFAI.cofai.models.vqfc import Dinov2VQFCCodec
import time
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def test_epoch(codec, vq_path):
    codec.eval()
    device = next(codec.parameters()).device
    codec.load_state_dict_compressor(torch.load(vq_path)["vqvae_state_dict"])

    eval_acc = 0.0
    eval_acc_ori = 0.0
    eval_mse = 0.0
    eval_rate = 0.0

    raw_dir = f"{PROJECT_ROOT}/features/vqfc/cls/test"
    with open(
        f"{PROJECT_ROOT}/examples/vqfc/cfg/imagenet_selected_label500.txt", "r"
    ) as f:
        data = f.readlines()

    num = 0
    enc_time_total = 0.0
    dec_time_total = 0.0

    for x in tqdm(data):
        file_name = x.split()[0]
        y = x.split()[1]
        batch_y = torch.tensor([int(y)]).to(device)

        feat_np = np.load(f"{raw_dir}/{file_name}.npy")
        feat_t = torch.from_numpy(feat_np).to(device)
        feat_in = feat_t.squeeze(0)  # [257,1536]

        with torch.no_grad():
            start_enc = time.time()
            coded_unit = codec.compress(feat_in)
            strings = coded_unit["strings"]["vqfc"][0]
            feat_shape = coded_unit["pstate"]["feat_shape"]
            end_enc = time.time()
            enc_time_total += end_enc - start_enc

            bit_stream_size = sum(len(s) for s in strings) * 8

            start_dec = time.time()
            feat_hat = codec.decompress(strings, feat_shape)
            end_dec = time.time()
            dec_time_total += end_dec - start_dec

            feat_recon = feat_hat.unsqueeze(0)
            feat_recon_npy = feat_recon.cpu().numpy()

            aug_feature_dq_list = [[feat_recon[0]]]
            aug_feature_dq_list_ori = [[feat_t[0]]]

            out_net = codec.forward_decode(aug_feature_dq_list[0])
            out_net_ori = codec.forward_decode(aug_feature_dq_list_ori[0])

            pred = torch.max(out_net, 1)[1]
            pred_ori = torch.max(out_net_ori, 1)[1]

            eval_acc += (pred == batch_y).sum().item()
            eval_acc_ori += (pred_ori == batch_y).sum().item()

            mse = (np.square(feat_np - feat_recon_npy)).mean()
            eval_mse += mse
            eval_rate += bit_stream_size
            num += 1

    eval_mse = eval_mse / num
    eval_acc = eval_acc / num
    eval_acc_ori = eval_acc_ori / num
    eval_rate = eval_rate / (num * 257 * 1536)

    print(
        f"\t=======MSE: {eval_mse:.6f}=======\n"
        f"\t=======rate: {eval_rate:.6f}=======\n"
        f"\t=====Eval Accuracy: {eval_acc:.6f}=====\n"
        f"\t==Original Feature Eval Accuracy: {eval_acc_ori:.6f}==\n"
    )
    print(f"Average Encoding Time per Image: {enc_time_total / num:.4f} s")
    print(f"Average Decoding Time per Image: {dec_time_total / num:.4f} s")
    print("=============================================\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument(
        "--vq_path",
        type=str,
        default=f"{PROJECT_ROOT}/weights/vqfc/cls/epoch_100num_8chunk_1.pth.tar",
    )
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=8)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--lmbda", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    codec = Dinov2VQFCCodec(
        vqfc_kwargs=dict(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            num_chunks=args.num_chunks,
            lmbda=args.lmbda,
        ),
        build_dino=True,
        dino_kwargs=dict(layers=1, pretrained=True),
        freeze_dino=True,
    ).to(device)

    test_epoch(codec, args.vq_path)


if __name__ == "__main__":
    main(sys.argv[1:])
