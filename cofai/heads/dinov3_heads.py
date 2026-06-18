import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from mpcompress.backbone.dinov3.eval.depth.metrics import calculate_depth_metrics


def _extract_linear_head_state_dict(ckpt_obj):
    """Extract linear-head state_dict with keys {'weight', 'bias'} from checkpoints.

    Supported inputs include:
    - direct linear state_dict: {'weight', 'bias'}
    - wrapped dicts: {'state_dict': ...} / {'model_state_dict': ...}
    - full-model state_dict with prefixes like:
      'linear_head.', 'module.linear_head.', 'head.linear_head.', 'module.head.linear_head.'
    """
    state_dict = ckpt_obj
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        return state_dict

    # Already a linear layer state dict.
    if "weight" in state_dict and "bias" in state_dict:
        return {"weight": state_dict["weight"], "bias": state_dict["bias"]}

    prefixes = (
        "linear_head.",
        "module.linear_head.",
        "head.linear_head.",
        "module.head.linear_head.",
        "head.",
        "module.head.",
    )
    for prefix in prefixes:
        w_key = f"{prefix}weight"
        b_key = f"{prefix}bias"
        if w_key in state_dict and b_key in state_dict:
            return {"weight": state_dict[w_key], "bias": state_dict[b_key]}

    return state_dict


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    """Resize input tensor using interpolation.

    Args:
        input (torch.Tensor): Input tensor to resize. Expected shape is (N, C, H, W).
        size (tuple[int, int], optional): Target size (height, width). Defaults to None.
        scale_factor (float or tuple[float, float], optional): Multiplier for spatial size.
            Defaults to None.
        mode (str): Interpolation mode. Options: 'nearest', 'bilinear', 'area', etc.
            Defaults to "nearest".
        align_corners (bool, optional): Whether to align corners. Defaults to None.
        warning (bool): Whether to show alignment warnings. Defaults to True.

    Returns:
        output (torch.Tensor): Resized tensor.
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class Dinov3SegmentationHead(torch.nn.Module):

    """Segmentation head for DINOv3 model.

    This head consists of BatchNorm and Conv layers to produce segmentation
    predictions from multi-level features. Supports various input transformation
    modes and sliding window inference for large images.
    """
    def __init__(
        self,
        in_channels,
        in_index,
        input_transform,
        channels,
        resize_factors=None,
        align_corners=False,
        num_classes=21,
        patch_size=16,
        num_register_tokens=4,
        token_hw=None,              # (H, W) grid of patch tokens (optional, can pass at predict time)
        dropout_ratio=0,
        checkpoint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.in_index = in_index
        self.input_transform = input_transform
        self.channels = channels
        self.resize_factors = resize_factors
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.token_hw = token_hw
        self.align_corners = align_corners

        self.bn = nn.SyncBatchNorm(channels)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

        if checkpoint is not None:
            checkpoint = os.path.expanduser(str(checkpoint))
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Head checkpoint not found: {checkpoint}. "
                    f"Set model.head.checkpoint to a valid file, or null to train from scratch."
                )

            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

            # 兼容:
            # 1) 纯 state_dict
            # 2) {"state_dict": ...}
            # 3) 训练checkpoint {"model_state_dict": ..., "epoch": ...}
            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            # 若是整模型参数，提取 head.* / module.head.* 到当前head
            if isinstance(state_dict, dict) and any(
                k.startswith("head.") or k.startswith("module.head.")
                for k in state_dict.keys()
            ):
                head_sd = {}
                for k, v in state_dict.items():
                    if k.startswith("module.head."):
                        head_sd[k[len("module.head."):]] = v
                    elif k.startswith("head."):
                        head_sd[k[len("head."):]] = v
                state_dict = head_sd

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if len(missing) or len(unexpected):
                warnings.warn(
                    f"[Dinov3SegmentationHead] checkpoint loaded with strict=False, "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

    def _infer_hw_from_token_count(self, n_patch: int, token_hw=None):
        """Infer (H, W) from patch token count if possible."""
        hw = token_hw if token_hw is not None else self.token_hw
        if hw is None:
            # Try square fallback
            s = int(n_patch ** 0.5)
            if s * s == n_patch:
                return (s, s)
            raise ValueError(
                "Cannot infer token grid size (H, W) from token count. "
                "Please pass token_hw=(H, W) to the head or to predict()/slide_predict()."
            )
        h, w = hw
        if h * w != n_patch:
            raise ValueError(f"token_hw={hw} mismatch: h*w={h*w} but n_patch={n_patch}.")
        return hw

    def _tokens_to_map(self, x, token_hw=None):
        """Convert tokens to feature map when needed.

        Accepts:
          - (B, C, H, W): returns as-is
          - (B, C): -> (B, C, 1, 1)
          - (B, N, C): DINOv3 tokens -> drop CLS+REG -> reshape to (B, C, H, W)
        """
        if isinstance(x, list):
            raise TypeError("Expected Tensor, got list. Flatten lists before calling _tokens_to_map.")

        if x.dim() == 4:
            return x
        if x.dim() == 2:
            return x[:, :, None, None]
        if x.dim() == 3:
            b, n, c = x.shape
            n_drop = 1 + int(self.num_register_tokens)  # CLS + REGs
            if n <= n_drop:
                raise ValueError(f"Token sequence too short: N={n}, need > {n_drop}.")
            patch_tokens = x[:, n_drop:, :]  # (B, N_patch, C)
            n_patch = patch_tokens.shape[1]
            h, w = self._infer_hw_from_token_count(n_patch, token_hw=token_hw)
            patch_tokens = patch_tokens.transpose(1, 2).contiguous()  # (B, C, N_patch)
            feat = patch_tokens.view(b, c, h, w)  # (B, C, H, W)
            return feat
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    def _transform_inputs(self, inputs, token_hw=None):
        """Transform inputs for decoder (adapted for DINOv3 token outputs)."""

        if self.input_transform == "resize_concat":                                            
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list

            # convert any token-like features to maps
            inputs = [self._tokens_to_map(x, token_hw=token_hw) for x in inputs]

            # select indices
            inputs = [inputs[i] for i in self.in_index]

            # optional per-level scaling (same as your original logic)
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (
                    len(self.resize_factors),
                    len(inputs),
                )
                inputs = [
                    resize(
                        input=x,
                        scale_factor=f,
                        mode="bilinear" if f >= 1 else "area",
                        align_corners=self.align_corners if (f >= 1) else None,
                    )
                    for x, f in zip(inputs, self.resize_factors)
                ]

            # upsample to same spatial size then concat on channel dim
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)

        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
            inputs = [self._tokens_to_map(x, token_hw=token_hw) for x in inputs]
            if len(inputs) == 1:
                inputs = inputs[0]
            else:
                upsampled_inputs = [
                    resize(
                        input=x,
                        size=inputs[0].shape[2:],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
                inputs = torch.cat(upsampled_inputs, dim=1)

        else:
            x = inputs[self.in_index]
            inputs = self._tokens_to_map(x, token_hw=token_hw)

        assert inputs.shape[1] == self.channels, (
            f"Input channels {inputs.shape[1]} does not match expected channels {self.channels}"
        )
        return inputs

    def forward(self, inputs, token_hw=None):
        x = self._transform_inputs(inputs, token_hw=token_hw)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv_seg(x)
        return x

    def predict(self, inputs, scale=1, size=None, token_hw=None):
        seg_logits = self.forward(inputs, token_hw=token_hw)
        _, _, tok_h, tok_w = seg_logits.shape
        if scale != 1:
            seg_logits = resize(
                input=seg_logits,
                size=(int(tok_h * scale), int(tok_w * scale)),
                mode="bilinear",
                align_corners=self.align_corners,
            )
        elif size is not None:
            assert isinstance(size, tuple)
            seg_logits = resize(
                input=seg_logits,
                size=size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return seg_logits

    def slide_predict(
        self,
        feature_list,
        current_size,
        slide_window,
        slide_stride,
        target_size=None,
        token_hw=None,
    ):
        device = next(self.conv_seg.parameters()).device
        h_img, w_img = current_size
        h_stride, w_stride = slide_stride
        h_crop, w_crop = slide_window
        batch_size = feature_list[0][0].shape[0]
        num_classes = self.num_classes

        preds = torch.zeros((batch_size, num_classes, h_img, w_img), device=device)
        count_mat = torch.zeros((batch_size, 1, h_img, w_img), device=device)

        i = 0
        for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_seg_logit = self.predict(
                    feature_list[i],
                    size=(h_crop, w_crop),
                    token_hw=token_hw,
                )
                preds += F.pad(
                    crop_seg_logit, (x1, preds.shape[3] - x2, y1, preds.shape[2] - y2)
                )
                count_mat[:, :, y1:y2, x1:x2] += 1
                i += 1

        assert (count_mat == 0).sum() == 0, "Zero count in count matrix detected"
        preds = preds / count_mat

        if target_size is not None:
            preds = resize(
                input=preds,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return preds


class Dinov3DepthHead(nn.Module):
    """Depth head compatible with the DINOv3 linear depth training checkpoint."""

    def __init__(
        self,
        in_channels,
        min_depth=0.001,
        max_depth=10.0,
        n_output_channels=256,
        use_batchnorm=True,
        use_backbone_norm=True,
        use_cls_token=False,
        checkpoint=None,
        align_corners=False,
        bins_strategy="linear",
        norm_strategy="linear",
        **kwargs,
    ):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = list(in_channels)
        self.channels = sum(self.in_channels)
        self.use_cls_token = bool(use_cls_token)
        if self.use_cls_token:
            self.channels *= 2
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.align_corners = align_corners
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy
        self.use_backbone_norm = bool(use_backbone_norm)

        self.batchnorm_layer = nn.SyncBatchNorm(self.channels) if use_batchnorm else nn.Identity()
        self.conv_depth = nn.Conv2d(self.channels, int(n_output_channels), kernel_size=1, padding=0, stride=1)
        nn.init.normal_(self.conv_depth.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv_depth.bias, 0)

        if checkpoint is not None:
            checkpoint = os.path.expanduser(str(checkpoint))
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"Head checkpoint not found: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state_dict = self._extract_depth_head_state_dict(ckpt)
            try:
                self.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                warnings.warn(
                    f"[Dinov3DepthHead] checkpoint loaded with strict=False, "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

    @staticmethod
    def _extract_depth_head_state_dict(ckpt_obj):
        state_dict = ckpt_obj
        if isinstance(state_dict, dict):
            if "linear_head" in state_dict:
                state_dict = state_dict["linear_head"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        if not isinstance(state_dict, dict):
            return state_dict

        normalized_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key.split("module.", 1)[1]
            if key.startswith("decoder."):
                key = key.split("decoder.", 1)[1]
            if key.startswith("linear_head."):
                key = key.split("linear_head.", 1)[1]
            if key.startswith("batchnorm."):
                key = key.replace("batchnorm.", "batchnorm_layer.", 1)
            if key.startswith("conv."):
                key = key.replace("conv.", "conv_depth.", 1)
            normalized_state_dict[key] = value
        return normalized_state_dict

    def _transform_inputs(self, inputs):
        inputs = list(inputs)
        feats = []
        for x in inputs:
            if self.use_cls_token:
                assert isinstance(x, (list, tuple)) and len(x) >= 2, "Missing class tokens"
                x, cls_token = x[0], x[1]
                if x.dim() == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                x = torch.cat((x, cls_token), dim=1)
            else:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                if x.dim() == 2:
                    x = x[:, :, None, None]
            feats.append(x)

        feats = [
            F.interpolate(
                input=x,
                size=feats[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for x in feats
        ]
        return torch.cat(feats, dim=1)

    def _features_to_depth(self, x):
        n_bins = x.shape[1]
        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device, dtype=x.dtype)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth, device=x.device, dtype=x.dtype)),
                    torch.log(torch.tensor(self.max_depth, device=x.device, dtype=x.dtype)),
                    n_bins,
                    device=x.device,
                    dtype=x.dtype,
                ).exp()
            else:
                raise ValueError(f"Unsupported bins_strategy: {self.bins_strategy}")

            if self.norm_strategy == "linear":
                logit = torch.relu(x) + 0.1
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(x, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(x)
                logit = logit / logit.sum(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported norm_strategy: {self.norm_strategy}")
            return torch.einsum("bchw,c->bhw", logit, bins).unsqueeze(1)
        return torch.relu(x) + self.min_depth

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.batchnorm_layer(x)
        x = self.conv_depth(x)
        x = self._features_to_depth(x)
        return x

    def predict(self, inputs, size=None):
        depth = self.forward(inputs)
        if size is not None and depth.shape[-2:] != size:
            depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=False)
        return depth
