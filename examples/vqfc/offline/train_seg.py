import os
import argparse
import random
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from dataset_seg import Dinov2DatasetTrain
from cofai.models.vqfc import Dinov2VQFCCodec
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
tb_logger = None


def parse_args(argv):
    p = argparse.ArgumentParser()

    # train
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--checkpoint", type=str, default="/output/")
    p.add_argument("--save", action="store_true", default=True)

    # vqfc
    p.add_argument("--embedding_dim", type=int, default=10)
    p.add_argument("--num_embeddings", type=int, default=32)
    p.add_argument("--num_chunks", type=int, default=32)
    p.add_argument("--lmbda", type=float, default=1.0)

    # validate (seg eval inside codec)
    p.add_argument(
        "--list_file", type=str, default=f"{PROJECT_ROOT}/examples/vqfc/cfg/val_100.txt"
    )
    p.add_argument("--img_root", type=str, default=f"{PROJECT_ROOT}/data/VOC2012")
    p.add_argument("--feat_dir", type=str, default=f"{PROJECT_ROOT}/features/seg/test")
    p.add_argument(
        "--feat_aug_dir", type=str, default=f"{PROJECT_ROOT}/features/seg/test"
    )
    p.add_argument("--head_dataset", type=str, default="voc2012")
    p.add_argument("--head_type", type=str, default="linear")
    p.add_argument("--num_classes", type=int, default=21)

    # feature format
    p.add_argument("--n_view", type=int, default=2)
    p.add_argument("--n_chan", type=int, default=1)
    p.add_argument("--feat_h", type=int, default=1370)
    p.add_argument("--feat_w", type=int, default=1536)

    return p.parse_args(argv)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(codec: Dinov2VQFCCodec, loader, optimizer, epoch: int):
    codec.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec.to(device)

    for batch_idx, batch in enumerate(loader):
        feat = batch[0].to(device)
        feat = torch.clamp(feat, min=-5, max=5)
        feat = feat.squeeze()  # keep your original behavior

        optimizer.zero_grad()
        feat_recon, mse_loss0, rd_loss, rate, encoding_inds = codec(feat)
        loss = rd_loss  # keep your original training objective
        loss.backward()
        optimizer.step()

        steps = epoch * len(loader) + batch_idx
        tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], steps)
        tb_logger.add_scalar("train_rd_loss", rd_loss.item(), steps)
        tb_logger.add_scalar("train_rate", rate.item(), steps)

        if batch_idx % 10 == 0:
            print(
                f"[train] epoch={epoch} step={batch_idx}/{len(loader)} rd={rd_loss.item():.4f} rate={rate.item():.4f}"
            )


@torch.no_grad()
def validate_epoch(codec: Dinov2VQFCCodec, args, epoch: int):
    # IMPORTANT: load_vq=False, evaluate current in-memory weights
    metrics = codec.seg_eval_from_feature_files(
        load_vq=False,
        vq_path=None,
        list_file=args.list_file,
        img_root=args.img_root,
        feat_dir=args.feat_dir,
        feat_aug_dir=args.feat_aug_dir,
        head_dataset=args.head_dataset,
        head_type=args.head_type,
        num_classes=args.num_classes,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    tb_logger.add_scalar("miou_ori", metrics["miou_ori"], epoch)
    tb_logger.add_scalar("miou_recon", metrics["miou_recon"], epoch)
    tb_logger.add_scalar("mse_ori", metrics["mse_ori"], epoch)
    tb_logger.add_scalar("mse_recon", metrics["mse_recon"], epoch)
    tb_logger.add_scalar("bpp_avg", metrics["bpp_avg"], epoch)

    print(f"[val] epoch={epoch} {metrics}")
    return metrics


def save_checkpoint(state, filename: str):
    torch.save(state, filename)


def main(argv):
    args = parse_args(argv)

    os.makedirs(args.checkpoint, exist_ok=True)

    global tb_logger
    tb_logger = SummaryWriter(os.path.join(args.checkpoint, "events"))

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    codec = Dinov2VQFCCodec(
        vqfc_kwargs=dict(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            num_chunks=args.num_chunks,
            lmbda=args.lmbda,
        )
    ).to(device)

    train_dataset = Dinov2DatasetTrain(train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    optimizer = torch.optim.Adam(codec.vqfc.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

    for epoch in range(1, args.epochs + 1):
        # validate first (same behavior as your old script)
        metrics = validate_epoch(codec, args, epoch)
        train_one_epoch(codec, train_loader, optimizer, epoch)
        scheduler.step()

        if args.save and epoch == args.epochs:
            ckpt_path = os.path.join(
                args.checkpoint,
                f"epoch_{args.epochs}num_{args.num_embeddings}chunk_{args.num_chunks}.pth.tar",
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "vqvae_state_dict": codec.state_dict_compressor(),
                    "mIOU": metrics["miou_recon"],
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
