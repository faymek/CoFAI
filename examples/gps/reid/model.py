import copy

import torch
import torch.nn as nn

from cofai.backbone.gps_transreid import build_gps_transreid_vit


def _make_bottleneck() -> nn.BatchNorm1d:
    bottleneck = nn.BatchNorm1d(768)
    bottleneck.bias.requires_grad_(False)
    return bottleneck


class _GPSCheckpointAdapter(nn.Module):
    """Construct and load only the released modules needed for evaluation."""

    def __init__(self, cfg, camera_num, view_num):
        super().__init__()
        model_cfg = cfg.model
        camera_num = camera_num if model_cfg.sie_camera else 0
        view_num = view_num if model_cfg.sie_view else 0

        self.base = build_gps_transreid_vit(
            img_size=model_cfg.image_size,
            sie_xishu=model_cfg.sie_coefficient,
            camera=camera_num,
            view=view_num,
            stride_size=model_cfg.stride_size,
            drop_path_rate=model_cfg.drop_path,
            drop_rate=model_cfg.dropout,
            attn_drop_rate=model_cfg.attention_dropout,
            pruning_layers=model_cfg.pruning_layers,
            pruning_ratios=model_cfg.pruning_ratios,
            propagation_max_iter=model_cfg.propagation_max_iter,
            beta=model_cfg.beta,
        )

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        self.bottleneck = _make_bottleneck()
        self.bottleneck_1 = _make_bottleneck()
        self.bottleneck_2 = _make_bottleneck()
        self.bottleneck_3 = _make_bottleneck()
        self.bottleneck_4 = _make_bottleneck()

    def load_checkpoint(self, trained_path):
        param_dict = torch.load(
            trained_path,
            map_location="cpu",
            weights_only=True,
        )
        if "model" in param_dict:
            param_dict = param_dict["model"]
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        checkpoint_state = {
            key.removeprefix("module."): value for key, value in param_dict.items()
        }
        model_state = self.state_dict()
        training_only_prefixes = (
            "classifier",
            "cam_classifier.",
            "scenario_classifier.",
        )
        known_training_only = {
            key for key in checkpoint_state if key.startswith(training_only_prefixes)
        }
        known_training_only.update(
            {
                "base.camera_tokens",
                "base.scenario_tokens",
            }
        )
        unexpected = set(checkpoint_state) - set(model_state)
        unsupported = unexpected - known_training_only
        shape_mismatches = {
            key: (tuple(checkpoint_state[key].shape), tuple(model_state[key].shape))
            for key in set(checkpoint_state) & set(model_state)
            if checkpoint_state[key].shape != model_state[key].shape
        }
        missing = set(model_state) - set(checkpoint_state)
        if unsupported or shape_mismatches or missing:
            raise RuntimeError(
                "checkpoint is incompatible with the GPS ReID model: "
                f"unexpected={sorted(unsupported)}, "
                f"shape_mismatches={shape_mismatches}, missing={sorted(missing)}"
            )

        compatible_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key not in known_training_only
        }
        self.load_state_dict(compatible_state, strict=True)
        ignored = sorted(unexpected & known_training_only)
        if ignored:
            print(f"Ignored known training-only checkpoint keys: {ignored}")
        print("Loading pretrained model from {}".format(trained_path))


def build_checkpoint_modules(cfg, camera_num, view_num):
    return _GPSCheckpointAdapter(
        cfg,
        camera_num,
        view_num,
    )
