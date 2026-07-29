import copy

import torch
import torch.nn as nn

from cofai.backbone.gps_transreid_vit import build_gps_transreid_vit


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)

    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class GPSCheckpointAdapter(nn.Module):
    """Construct and load only the released modules needed for evaluation."""

    def __init__(self, camera_num, view_num, cfg, rearrange):
        super().__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.cls_token_num = cfg.cls_token_num

        print(
            "using Transformer_type: {} as a backbone".format(
                cfg.MODEL.TRANSFORMER_TYPE
            )
        )

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = build_gps_transreid_vit(
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            cls_token_num=cfg.cls_token_num,
            pruning_layers=cfg.MODEL.BACKGROUND_PRUNING_LAYERS,
            pruning_ratios=cfg.MODEL.BACKGROUND_PRUNING_RATIOS,
            propagation_max_iter=cfg.MODEL.PROPAGATION_MAX_ITER,
            beta=cfg.MODEL.BETA,
        )

        if pretrain_choice != "none" or model_path:
            raise ValueError(
                "GPS evaluation loads the task checkpoint after construction; "
                "an additional backbone pretrain checkpoint is unsupported"
            )

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print("using shuffle_groups size:{}".format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print("using shift_num size:{}".format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print("using divide_length size:{}".format(self.divide_length))
        self.rearrange = rearrange

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
    if (
        cfg.MODEL.NAME != "transformer"
        or not cfg.MODEL.JPM
        or cfg.MODEL.TRANSFORMER_TYPE != "vit_base_patch16_224_TransReID_seq"
    ):
        raise ValueError(
            "GPS evaluation requires the JPM-enabled "
            "vit_base_patch16_224_TransReID_seq model"
        )
    model = GPSCheckpointAdapter(
        camera_num,
        view_num,
        cfg,
        rearrange=cfg.MODEL.RE_ARRANGE,
    )
    print("===========building transformer with JPM module ===========")
    return model
