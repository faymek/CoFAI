import argparse
import random
import sys
import numpy as np
import torch
from cofai.models.vqfc import Dinov2VQFCCodec
import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Segmentation eval (VQFC)")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--seed", type=int, default=0, help="Set random seed")
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_embeddings", type=int, default=128)
    parser.add_argument(
        "--vq_path",
        type=str,
        default=f"{PROJECT_ROOT}/weights/vqfc/seg/epoch_100num_128dim_16chunk_1.pth.tar",
    )
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--lmbda", type=float, default=1.0)

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    codec = Dinov2VQFCCodec(
        vqfc_kwargs=dict(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            num_chunks=args.num_chunks,
            lmbda=args.lmbda,
        ),
        build_dino=False,
        freeze_dino=True,
    ).to(device)

    metrics = codec.seg_eval_from_feature_files(
        vq_path=args.vq_path,
        device=device,
    )

    print(f"\t=======Ori-mIOU======: {metrics['miou_ori']}")
    print(f"\t=======Ori-MSE======: {metrics['mse_ori']}")
    print(f"\t=======Recon-mIOU======: {metrics['miou_recon']}")
    print(f"\t=======Recon-MSE======: {metrics['mse_recon']}")
    print(f"Average Data Size (in bpp): {metrics['bpp_avg']}")


if __name__ == "__main__":
    main(sys.argv[1:])
