import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CoFAI.examples.vitdet.hyper_token_codec import HyperVitTokenCodec

load_dotenv()
project_root = os.environ.get("PROJECT_ROOT")


def evaluate(config):
    # === 加载模型 ===
    model = HyperVitTokenCodec(feat_dims=768).to(config["device"])
    checkpoint = torch.load(config["model_path"], map_location=config["device"])
    model.load_state_dict(checkpoint)
    model.eval()
    model.update()
    # === 保存重建特征 ===
    rec_dir = f"/home/faymek/compression_vit/vit-b_layer{config['layer']}/rec"
    os.makedirs(rec_dir, exist_ok=True)
    # === 遍历该层的所有 .npy 文件 ===
    files = sorted(
        [f for f in os.listdir(config["feature_dir"]) if f.endswith(".npy")],
        key=lambda x: int(x.split(".")[0]),
    )
    file_paths = [os.path.join(config["feature_dir"], f) for f in files]

    layer_name = f"layer{config['layer']}"
    stats = {"count": 0, "orig": 0, "comp": 0, "mse": 0.0}

    for np_path in tqdm(file_paths, desc=f"Evaluating {layer_name}"):
        feat_np = np.load(np_path)
        feat_tensor = torch.from_numpy(feat_np).float().to(config["device"])
        orig_size = os.path.getsize(np_path)

        with torch.no_grad():
            out = model.compress(feat_tensor)
            strings = out["strings"]
            shape = out["shape"]
            comp_size = len(strings[0][0]) + sum(len(s) for s in strings[1])
            recon = model.decompress(strings, shape)["x_hat"]
            mse = torch.mean((recon - feat_tensor) ** 2).item()
            np.save(
                os.path.join(rec_dir, f"{stats['count']}.npy"),
                recon.detach().cpu().numpy(),
            )

        stats["count"] += 1
        stats["orig"] += orig_size
        stats["comp"] += comp_size
        stats["mse"] += mse

    # === 输出结果 ===
    avg_orig = stats["orig"] / stats["count"]
    avg_comp = stats["comp"] / stats["count"]
    avg_mse = stats["mse"] / stats["count"]
    compression_ratio = avg_comp / avg_orig

    print(f"\n📊 Evaluation Result for {layer_name}")
    print(f"  Test samples             : {stats['count']}")
    print(f"  Avg original size        : {avg_orig:.2f} bytes")
    print(f"  Avg compressed size      : {avg_comp:.2f} bytes")
    print(f"  Avg compression ratio    : {compression_ratio:.6f}x")
    print(f"  Avg MSE                  : {avg_mse:.6f}")


def get_config(layer):
    feature_dir = f"{project_root}/features/vitdet/vit-b_layer{layer}/npy"
    save_dir = f"{project_root}/weights/vitdet/checkpoints/layer{layer}"

    return {
        "layer": layer,
        "feature_dir": feature_dir,
        "model_path": f"{save_dir}/model_final.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer", type=int, required=True, help="Layer index (e.g. 3, 6, 9, 12)"
    )
    args = parser.parse_args()

    config = get_config(args.layer)
    print("Config:")
    for k, v in config.items():
        print(f"{k}: {v}")
    evaluate(config)
