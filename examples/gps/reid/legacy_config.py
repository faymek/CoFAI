"""Translate the compact example YAML into the original model interface."""

from __future__ import annotations

from types import SimpleNamespace


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_ns(item) for item in value]
    return value


def build_legacy_config(cfg, checkpoint: str | None = None):
    model = cfg.model
    data = cfg.dataset
    evaluation = cfg.evaluation
    return _ns(
        {
            "descrip": "gps_multi_view",
            "cls_token_num": 1,
            "MODEL": {
                # Evaluation immediately loads the task checkpoint, so requiring a
                # second ImageNet checkpoint only makes the documented eval inputs
                # incomplete.
                "PRETRAIN_CHOICE": "none",
                "PRETRAIN_PATH": "",
                "METRIC_LOSS_TYPE": "triplet",
                "ID_LOSS_TYPE": "none",
                "IF_LABELSMOOTH": "off",
                "IF_WITH_CENTER": "no",
                "NAME": "transformer",
                "NO_MARGIN": True,
                "TRANSFORMER_TYPE": "vit_base_patch16_224_TransReID_seq",
                "STRIDE_SIZE": list(model.stride_size),
                "SIE_CAMERA": bool(model.sie_camera),
                "SIE_VIEW": bool(model.sie_view),
                "SIE_COE": float(model.sie_coefficient),
                "JPM": True,
                "SHIFT_NUM": 8,
                "SHUFFLE_GROUP": 2,
                "DEVIDE_LENGTH": 4,
                "RE_ARRANGE": True,
                "TOKEN_PRUNING_RATIO": 0.0,
                "BACKGROUND_PRUNING_LAYERS": list(model.pruning_layers),
                "BACKGROUND_PRUNING_RATIOS": list(model.pruning_ratios),
                "PROPAGATION_MAX_ITER": int(model.propagation_max_iter),
                "BETA": float(model.beta),
                "DROP_PATH": float(model.drop_path),
                "DROP_OUT": float(model.dropout),
                "ATT_DROP_RATE": float(model.attention_dropout),
                "DIST_TRAIN": False,
                "COS_LAYER": False,
                "NECK": "bnneck",
            },
            "INPUT": {
                "SIZE_TRAIN": list(model.image_size),
                "SIZE_TEST": list(model.image_size),
                "PROB": 0.5,
                "RE_PROB": 0.5,
                "PADDING": 10,
                "PIXEL_MEAN": list(model.pixel_mean),
                "PIXEL_STD": list(model.pixel_std),
            },
            "DATASETS": {
                "NAMES": str(data.name),
                "ROOT_DIR": str(cfg.data_root),
            },
            "DATALOADER": {
                "NUM_WORKERS": int(evaluation.workers),
                "SAMPLER": "softmax",
            },
            "SOLVER": {"IMS_PER_BATCH": int(evaluation.batch_size)},
            "TEST": {
                "IMS_PER_BATCH": int(evaluation.batch_size),
                "FEAT_NORM": "yes" if evaluation.feature_normalize else "no",
                "NECK_FEAT": str(model.neck_feature),
                "RE_RANKING": bool(evaluation.reranking),
                "WEIGHT": checkpoint or str(cfg.checkpoint),
            },
        }
    )
