"""
Models with no compression, just for testing purposes.

This module provides bypass models that perform feature extraction and task inference
without any compression. These models are useful for testing and benchmarking the
feature extraction pipeline.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from compressai.registry import register_model
from compressai.models.base import CompressionModel
from cofai.backbone import *
from cofai.engine.registry import instantiate_class


@register_model("Dinov2TimmBypass")
class Dinov2TimmBypass(CompressionModel):
    """
    A bypass model using DINOv2-Timm backbone for feature extraction without compression.

    This model performs feature extraction using a DINOv2-Timm backbone and supports
    multiple downstream tasks (classification, segmentation) without any compression
    operations. It returns empty byte strings as a placeholder for compressed data.

    Args:
        dino_backbone (dict): Configuration dictionary for the DINOv2-Timm backbone.
            Passed directly to Dinov2TimmBackbone constructor.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        dino (Dinov2TimmBackbone): The DINOv2-Timm backbone model.
        patch_size (int): Patch size used by the backbone model.
    """

    def __init__(
        self,
        dino_backbone={},
        heads: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.patch_size = self.dino.patch_size
        self.heads = nn.ModuleDict()
        if heads:
            if not isinstance(heads, dict):
                raise TypeError("heads must be a dict mapping task -> head config")
            for task, hcfg in heads.items():
                if not isinstance(task, str):
                    continue
                if not isinstance(hcfg, dict) or "type" not in hcfg:
                    raise ValueError(f"heads.{task} must be a dict with a 'type' field")
                self.heads[task] = instantiate_class(hcfg)

    def forward_test(self, x, tasks=[], **kwargs):
        """
        Forward pass for testing/inference without compression.

        Extracts features using the DINOv2 backbone and generates task-specific
        features (classification, segmentation) without performing any compression.
        Returns empty byte strings as a placeholder for compressed data.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            tasks (list of str): ``cls``, ``semseg``, ``rae`` — same head rules as ``MPC_I2.forward_test``.
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_unit (dict): Dictionary containing:

                - "strings": Dictionary with "bypass" key containing empty bytes
                - "pstate": Dictionary with "token_res" (token resolution)

            task_feats (dict): Per-task outputs when the corresponding head is configured (raw patch tokens for ``rae`` if no decoder head).

        """
        with torch.inference_mode():
            h_dino = self.dino.encode(x)

            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            task_feats = {}
            if "cls" in tasks:
                feat_cls = self.dino.decode_cls(h_dino)
                if "cls" in self.heads:
                    task_feats["cls"] = self.heads["cls"](feat_cls)
            if "semseg" in tasks:
                feat_semseg = self.dino.decode_seg(h_dino, token_res)
                if "semseg" in self.heads:
                    task_feats["semseg"] = self.heads["semseg"].predict(
                        feat_semseg, scale=int(self.patch_size)
                    )
            if "rae" in tasks:
                feat_rae = self.dino.decode_rae(h_dino, token_res)
                head_key = "rec" if "rec" in self.heads else ("rae" if "rae" in self.heads else None)
                if head_key is not None:
                    task_feats["rae"] = self.heads[head_key].predict(
                        feat_rae,
                        token_res=token_res,
                        token_format="patch",
                    )
                else:
                    task_feats["rae"] = feat_rae

            coded_unit = {
                "strings": {"bypass": [[b""]]},  # empty bytes
                "pstate": {"token_res": token_res},
            }
            return coded_unit, task_feats

    def get_feature_numel(self, x):
        """
        Calculate the total number of elements in the extracted features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            numel (int): Total number of elements in the feature tensor.
        """
        h_dino = self.dino.encode(x)
        return h_dino.numel()


@register_model("Dinov3TimmBypass")
class Dinov3TimmBypass(CompressionModel):
    """
    A bypass model using DINOv3-Timm backbone for feature extraction without compression.

    This model performs feature extraction using a DINOv3-Timm backbone and supports
    multiple downstream tasks (classification, segmentation, depth estimation) without
    any compression operations. It returns empty byte strings as a placeholder for
    compressed data.

    Args:
        dino_backbone (dict): Configuration dictionary for the DINOv3-Timm backbone.
            Passed directly to Dinov3TimmBackbone constructor.
        heads (dict, optional): Mapping of task name to head configuration.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        dino (Dinov3TimmBackbone): The DINOv3-Timm backbone model.
        patch_size (int): Patch size used by the backbone model.
        heads (nn.ModuleDict): Task-specific head modules.
    """

    def __init__(
        self,
        dino_backbone={},
        heads: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        if "type" in dino_backbone:
            self.dino = instantiate_class(dino_backbone)
        else:
            self.dino = Dinov3TimmBackbone(**dino_backbone)
        self.patch_size = self.dino.patch_size

        # Optional task heads (engine-native path). When a head is configured for a
        # task label, ``forward_test`` returns the final task output under that label
        # (``semseg`` / ``depth``); otherwise it returns raw backbone features under
        # ``seg`` / ``depth`` (legacy ``run_eval.py`` applies heads externally).
        self.heads = nn.ModuleDict()
        if heads:
            if not isinstance(heads, dict):
                raise TypeError("heads must be a dict mapping task -> head config")
            for task, hcfg in heads.items():
                if not isinstance(task, str) or hcfg is None:
                    continue
                if not isinstance(hcfg, dict) or "type" not in hcfg:
                    raise ValueError(f"heads.{task} must be a dict with a 'type' field")
                self.heads[task] = instantiate_class(hcfg)

    def forward_test(self, x, qp=0, tasks=[], **kwargs):
        """
        Forward pass for testing/inference without compression.

        Extracts features with the DINOv3 backbone and produces raw task features
        (``cls`` / ``seg`` / ``depth``) without any compression. The ``qp`` argument
        is accepted for interface compatibility and ignored.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Unused; kept for a uniform codec interface.
            tasks (list of str): Subset of ``cls`` / ``seg`` / ``depth``.

        Returns:
            coded_unit (dict): ``strings`` with an empty bypass payload and
                ``pstate`` with the token resolution.
            task_feats (dict): Per-task raw decoded features.
        """
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )

            task_feats = {}
            if "cls" in tasks:
                task_feats["cls"] = self.dino.decode_cls(h_dino)
            if "seg" in tasks:
                task_feats["seg"] = self.dino.decode_seg(h_dino, token_res)
            if "semseg" in tasks:
                feat = self.dino.decode_seg(h_dino, token_res)
                task_feats["semseg"] = self.heads["semseg"].predict(
                    feat, scale=int(self.patch_size)
                )
            if "depth" in tasks:
                feat = self.dino.decode_depth(h_dino, token_res)
                size = (
                    int(token_res[0]) * int(self.patch_size),
                    int(token_res[1]) * int(self.patch_size),
                )
                task_feats["depth"] = self.heads["depth"].predict(feat, size=size)

            coded_unit = {
                "strings": {"bypass": [[b""]]},  # empty bytes
                "pstate": {"token_res": token_res},
            }
            return coded_unit, task_feats

    def get_feature_numel(self, x):
        """
        Calculate the total number of elements in the extracted features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            numel (int): Total number of elements in the feature tensor.
        """
        h_dino = self.dino.encode(x)
        return h_dino.numel()
