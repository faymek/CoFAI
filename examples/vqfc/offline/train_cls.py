import os
import argparse
import random
import sys
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from cofai.models.vqfc import Dinov2VQFCCodec
from dataset_cls import Dinov2DatasetTrain
from tqdm import tqdm
from dotenv import load_dotenv
tb_logger = None
load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def train_one_epoch(codec, loss_functioner, train_loader, optimizer, epoch):
    device = "cuda"
    codec.train()
    codec.to(device)

    for batch_idx, batch in enumerate(train_loader):
        feat = batch[0].squeeze().to(device)  # [N,257,1536]
        feat = torch.clamp(feat, min=-10, max=10)

        optimizer.zero_grad()
        feat_recon, mse_loss, rd_loss, rate, encoding_inds = codec(feat)

        mse_loss = loss_functioner(feat_recon, feat)
        loss = rd_loss

        loss.backward()
        optimizer.step()

        steps = epoch * len(train_loader) + batch_idx
        if steps % 1 == 0:
            tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], steps)
            tb_logger.add_scalar("Train mse", mse_loss, steps)
            tb_logger.add_scalar("Train rate", rate, steps)
            tb_logger.add_scalar("Train loss", loss, steps)

            print(
                f"\tTrain epoch {epoch}"
                f"\tTrain batch {batch_idx}: ["
                f"\t{batch_idx * train_loader.batch_size}/{len(train_loader.dataset)}"
                f"\t({100.0 * batch_idx / len(train_loader):.0f}%)]\n"
                f"\tTrain mse loss: {mse_loss.item():.4f} |\n"
                f"\tTrain rate: {rate.item():.4f} |\n"
                f"\tTrain rd loss: {rd_loss.item():.4f} |"
            )


def validate_epoch(epoch, loss_functioner, codec):
    device = "cuda"
    codec.to(device)
    codec.eval()

    eval_acc = 0.0
    eval_acc_ori = 0.0
    eval_mse = 0.0
    eval_rate = 0.0

    raw_dir = f"{PROJECT_ROOT}/features/vqfc/cls/test"
    with open(
        f"{PROJECT_ROOT}/examples/vqfc/cfg/imagenet_selected_label500.txt", "r"
    ) as f:
        data = f.readlines()

    with torch.no_grad():
        num = 0
        for x in tqdm(data):
            file_name = x.split()[0]
            y = x.split()[1]
            batch_y = torch.tensor([int(y)]).to(device)

            aug_feature_dq_list_numpy = np.load(f"{raw_dir}/{file_name}.npy")
            aug_feature_dq_list_tensor = torch.from_numpy(aug_feature_dq_list_numpy).to(
                device
            )
            feat_in = aug_feature_dq_list_tensor.squeeze(0)  # [257,1536]

            feat_recon_squeeze, mse_loss, rd_loss, rate, encoding_inds = codec(feat_in)

            feat_recon = feat_recon_squeeze.unsqueeze(0)
            feat_recon_npy = feat_recon.cpu().numpy()

            aug_feature_dq_list = [[feat_recon[0]]]
            aug_feature_dq_list_ori = [[aug_feature_dq_list_tensor[0]]]

            out_net = codec.forward_decode(aug_feature_dq_list[0])
            out_net_ori = codec.forward_decode(aug_feature_dq_list_ori[0])

            pred = torch.max(out_net, 1)[1]
            pred_ori = torch.max(out_net_ori, 1)[1]

            eval_acc += (pred == batch_y).sum().item()
            eval_acc_ori += (pred_ori == batch_y).sum().item()

            org_feat = np.load(f"{raw_dir}/{file_name}.npy")
            mse = (np.square(org_feat - feat_recon_npy)).mean()
            eval_mse += mse
            eval_rate += rate
            num += 1

    eval_mse = eval_mse / num
    eval_acc = eval_acc / num
    eval_acc_ori = eval_acc_ori / num
    eval_rate = eval_rate / num

    step = epoch
    tb_logger.add_scalar("eval_mse", eval_mse, step)
    tb_logger.add_scalar("eval_acc", eval_acc, step)
    tb_logger.add_scalar("eval_acc_ori", eval_acc_ori, step)
    tb_logger.add_scalar("eval_rate", eval_rate, step)

    print("len test dataset: ", num)
    print(
        f"\t=======MSE: {eval_mse:.6f}=======\n"
        f"\t=======rate: {eval_rate:.6f}=======\n"
        f"\t=====Eval Accuracy: {eval_acc:.6f}=====\n"
        f"\t==Original Feature Eval Accuracy: {eval_acc_ori:.6f}==\n"
    )

    return eval_mse, eval_acc, eval_acc_ori


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--lr", default=1e-3, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--checkpoint", type=str, default=f"{PROJECT_ROOT}/runs/vqfc/cls/"
    )
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_embeddings", type=int, default=2)
    parser.add_argument("--num_chunks", type=int, default=32)
    parser.add_argument("--lmbda", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    global tb_logger
    tb_logger = SummaryWriter(os.path.join(args.checkpoint, "events"))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    device = "cuda"

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

    train_dataset = Dinov2DatasetTrain()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(codec.vqfc.parameters(), lr=args.lr)
    train_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    loss_functioner = torch.nn.MSELoss()

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    for epoch in range(1, args.epochs + 1):
        if optimizer.param_groups[0]["lr"] < 1e-6:
            optimizer.param_groups[0]["lr"] = 1e-6
            print("=======set min lr to 1e-6========")

        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_one_epoch(codec, loss_functioner, train_loader, optimizer, epoch)
        eval_mse, eval_acc, eval_acc_ori = validate_epoch(epoch, loss_functioner, codec)

        train_scheduler.step()

        if args.save and epoch == args.epochs:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "vqvae_state_dict": codec.state_dict_compressor(),
                    "acc": eval_acc,
                    "optimizer": optimizer.state_dict(),
                },
                True,
                filename=args.checkpoint
                + "epoch_"
                + str(epoch)
                + "num_"
                + str(args.num_embeddings)
                + "chunk_"
                + str(args.num_chunks)
                + ".pth.tar",
            )
            print(
                f"\tTest ACC:{eval_acc:.4f}|"
                f"\tTest ACC ORI:{eval_acc_ori:.4f}|"
                f"\tsave last epoch model of epoch: {epoch}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
