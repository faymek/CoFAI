import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyper_token_codec import HyperVitTokenCodec

load_dotenv()
project_root = os.environ.get("PROJECT_ROOT")
# save_dir = f"{project_root}/features/vitdet/vit-b_layer{layer_idx}/npy"


# === 单层特征数据集定义 ===
class SingleLayerFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = sorted(
            [f for f in os.listdir(feature_dir) if f.endswith(".npy")],
            key=lambda x: int(x.split(".")[0]),
        )
        self.file_paths = [os.path.join(feature_dir, f) for f in self.files]
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        feat = np.load(path)
        return torch.from_numpy(feat).float(), self.feature_dir  # 返回路径方便统计


def train(config):
    os.makedirs(config["save_dir"], exist_ok=True)
    dataset = SingleLayerFeatureDataset(config["feature_dir"])
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )

    model = HyperVitTokenCodec(feat_dims=768).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # === 初始化统计 ===
    layer_name = f"layer{config['layer']}"
    layer_stats = {layer_name: {"count": 0, "bpp_sum": 0.0, "mse_sum": 0.0}}

    step = 0
    epoch = 0

    with tqdm(total=config["max_iters"], desc=f"Training {layer_name}") as pbar:
        while step < config["max_iters"]:
            epoch += 1
            for batch, folders in dataloader:
                if step >= config["max_iters"]:
                    break

                batch = batch.to(config["device"]).squeeze(1)  # [B, 768, 64, 64]
                model.train()
                optimizer.zero_grad()
                bpp, mse, recon = model(batch)
                loss = bpp + config["lambda"] * mse
                loss.backward()
                optimizer.step()

                # 累计统计
                layer_stats[layer_name]["count"] += 1
                layer_stats[layer_name]["bpp_sum"] += bpp.item()
                layer_stats[layer_name]["mse_sum"] += mse.item()

                if step % 1000 == 0:
                    tqdm.write(
                        f"[Iter {step}] Loss: {loss.item():.6f} | BPP: {bpp.item():.6f} | MSE: {mse.item():.6f}"
                    )
                    avg_bpp = (
                        layer_stats[layer_name]["bpp_sum"]
                        / layer_stats[layer_name]["count"]
                    )
                    avg_mse = (
                        layer_stats[layer_name]["mse_sum"]
                        / layer_stats[layer_name]["count"]
                    )
                    tqdm.write(
                        f"📊 {layer_name}: Avg BPP={avg_bpp:.5f} | Avg MSE={avg_mse:.6f}"
                    )

                # if step % config["save_every"] == 0 and step != 0:
                #     save_path = os.path.join(config["save_dir"], f"model_iter_{step}.pt")
                #     torch.save(model.state_dict(), save_path)
                #     tqdm.write(f"✅ Saved model at iteration {step} → {save_path}")

                step += 1
                pbar.update(1)

    final_path = os.path.join(config["save_dir"], "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"🎉 Training complete. Final model saved to {final_path}")


def get_config(layer, lambda_weight):
    feature_dir = f"{project_root}/features/vitdet/vit-b_layer{layer}/npy"
    save_dir = f"{project_root}/weights/vitdet/checkpoints_new/layer{layer}"

    return {
        "feature_dir": feature_dir,
        "batch_size": 8,
        "lr": 2e-5,
        "lambda": lambda_weight,
        "max_iters": 40000,
        "save_every": 10000,
        "save_dir": save_dir,
        "layer": layer,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambda", type=float, required=True, help="Weight for MSE loss"
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Layer index (e.g. 3, 6, 9, 12)"
    )
    args = parser.parse_args()

    config = get_config(args.layer, args.__dict__["lambda"])
    print("Config:")
    for k, v in config.items():
        print(f"{k}: {v}")
    train(config)
