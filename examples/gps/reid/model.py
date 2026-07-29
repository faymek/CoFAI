import torch
import torch.nn as nn
import copy

from .backbone.vit_pytorch import vit_base_patch16_224_TransReID_seq
from .backbone.utils import shuffle_unit


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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
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

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            cfg=cfg,
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

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE != "none":
            raise ValueError(
                "GPS integration supports the released evaluation head only"
            )
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

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

    def encode(
        self,
        x,
        label=None,
        cam_label=None,
        view_label=None,
        multi_view=False,
        flip_view=False,
        extra_token=False,
        dataset_name="train",
        **kwargs,
    ):
        """Run the GPS-enabled encoder and expose the feature coding boundary."""
        if multi_view:
            features, lsort, flops, selection = self.base(
                x,
                cam_label=cam_label,
                view_label=view_label,
                label=label,
                multi_view=multi_view,
                flip_view=flip_view,
                extra_token=extra_token,
                dataset_name=dataset_name,
            )
            label = label[lsort]
        else:
            features = self.base(
                x,
                cam_label=cam_label,
                view_label=view_label,
                label=label,
                multi_view=multi_view,
                extra_token=extra_token,
                dataset_name=dataset_name,
            )
            lsort = None
            flops = 0.0
            selection = None
        return {
            "h": features,
            "context": {
                "label": label,
                "order": lsort,
                "flops": flops,
                "extra_token": bool(extra_token),
            },
            "selection": selection,
        }

    def decode(self, h, *, context=None, tasks=None, **kwargs):
        """Run the unchanged GPS ReID decoder and embedding heads."""
        features = h
        context = dict(context or {})
        label = context.get("label")
        lsort = context.get("order")
        flops = context.get("flops", 0.0)
        extra_token = bool(context.get("extra_token", False))
        bs = features.shape[0]
        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        aux_logits = []
        if extra_token:
            global_feat = b1_feat[:, 0 : self.cls_token_num].reshape(bs, -1)
            cls_token_num = self.cls_token_num
            bottle_global_feat = b1_feat[:, 0]
        else:
            global_feat = b1_feat[:, 0]
            cls_token_num = 1
            bottle_global_feat = global_feat

        # JPM branch
        feature_length = features.size(1) - cls_token_num
        patch_length = feature_length // self.divide_length
        token = features[:, 0:cls_token_num]

        if self.rearrange:
            x = shuffle_unit(
                features, self.shift_num, self.shuffle_groups, cls_token_num
            )
        else:
            x = features[:, cls_token_num:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length : patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]  # .reshape(bs,-1)

        # lf_3
        b3_local_feat = x[:, patch_length * 2 : patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]  # .reshape(bs,-1)

        # lf_4
        b4_local_feat = x[:, patch_length * 3 : patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]  # .reshape(bs,-1)

        feat = self.bottleneck(bottle_global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ("arcface", "cosface", "amsoftmax", "circle"):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)

            return (
                [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4],
                [
                    global_feat,
                    b1_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1),
                    b2_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1),
                    b3_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1),
                    b4_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1),
                ],
                lsort,
                flops,
                aux_logits,
            )  # global feature for triplet loss

            # return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
            #        [global_feat, b1_local_feat[:, 0:self.cls_token_num].reshape(bs,-1), \
            #                      b2_local_feat[:, 0:self.cls_token_num].reshape(bs,-1), \
            #                      b3_local_feat[:, 0:self.cls_token_num].reshape(bs,-1), \
            #                      b4_local_feat[:, 0:self.cls_token_num].reshape(bs,-1)],\
            #        lsort, flops
        else:
            if self.neck_feat == "after":
                return torch.cat(
                    [
                        feat,
                        local_feat_1_bn / 4,
                        local_feat_2_bn / 4,
                        local_feat_3_bn / 4,
                        local_feat_4_bn / 4,
                    ],
                    dim=1,
                ), lsort  # , [cam_logits, scenario_logits]
            else:
                return torch.cat(
                    [
                        global_feat,
                        b1_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1) / 4,
                        b2_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1) / 4,
                        b3_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1) / 4,
                        b4_local_feat[:, 0 : self.cls_token_num].reshape(bs, -1) / 4,
                    ],
                    dim=1,
                ), lsort  # , [cam_logits, scenario_logits]

    def forward(
        self,
        x,
        label=None,
        cam_label=None,
        view_label=None,
        multi_view=False,
        flip_view=False,
        extra_token=False,
        dataset_name="train",
    ):
        encoded = self.encode(
            x,
            label=label,
            cam_label=cam_label,
            view_label=view_label,
            multi_view=multi_view,
            flip_view=flip_view,
            extra_token=extra_token,
            dataset_name=dataset_name,
        )
        return self.decode(encoded["h"], context=encoded["context"])

    def load_param(self, trained_path):
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
        known_training_only = {
            "base.camera_tokens",
            "base.scenario_tokens",
            "cam_classifier.weight",
            "scenario_classifier.weight",
        }
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


__factory_T_type = {
    "vit_base_patch16_224_TransReID_seq": vit_base_patch16_224_TransReID_seq
}


def make_model(cfg, num_class, camera_num, view_num):
    if (
        cfg.MODEL.NAME != "transformer"
        or not cfg.MODEL.JPM
        or cfg.MODEL.TRANSFORMER_TYPE not in __factory_T_type
    ):
        raise ValueError(
            "GPS evaluation requires the JPM-enabled "
            "vit_base_patch16_224_TransReID_seq model"
        )
    model = build_transformer_local(
        num_class,
        camera_num,
        view_num,
        cfg,
        __factory_T_type,
        rearrange=cfg.MODEL.RE_ARRANGE,
    )
    print("===========building transformer with JPM module ===========")
    return model
