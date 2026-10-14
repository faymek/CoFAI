"""Plain-DETR detection head: loads DETR weights only; timm supplies 4-scale taps."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from .plain_detr.decoder import GlobalDecoder, GlobalDecoderLayer
from .plain_detr.model import PlainDETR, PostProcess
from .plain_detr.transformer import Transformer
from .plain_detr.utils import NestedTensor, inverse_sigmoid


class Dinov3PlainDETRHead(nn.Module):
    """Wrap Plain-DETR transformer / box head; features come from CTC timm taps."""

    def __init__(
        self,
        checkpoint: str,
        score_thr: float = 0.001,
        topk: int = 100,
    ):
        super().__init__()
        self.score_thr = float(score_thr)
        decoder_layer = GlobalDecoderLayer(
            d_model=768,
            d_ffn=2048,
            dropout=0.0,
            n_heads=8,
            norm_type="pre_norm",
            rpe_hidden_dim=512,
            rpe_type="linear",
            feature_stride=16,
        )
        decoder = GlobalDecoder(
            decoder_layer,
            num_layers=6,
            return_intermediate=True,
            look_forward_twice=True,
            d_model=768,
            norm_type="pre_norm",
        )
        transformer = Transformer(
            decoder,
            d_model=768,
            nhead=8,
            num_feature_levels=1,
            two_stage=True,
            two_stage_num_proposals=600,
            mixed_selection=True,
            norm_type="pre_norm",
            proposal_feature_levels=4,
            proposal_in_stride=16,
            proposal_tgt_strides=[8, 16, 32, 64],
            proposal_min_size=50,
            add_transformer_encoder=True,
            num_encoder_layers=6,
        )
        model = PlainDETR(
            transformer,
            in_channels=4 * 1024,
            num_classes=91,
            num_feature_levels=1,
            with_box_refine=True,
            two_stage=True,
            num_queries_one2one=300,
            num_queries_one2many=300,
            mixed_selection=True,
        )
        ckpt_path = os.path.expanduser(str(checkpoint))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Plain-DETR checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        filtered = {
            k: v
            for k, v in state.items()
            if not str(k).startswith("backbone.0.backbone")
        }
        model.load_state_dict(filtered, strict=True)
        model.eval()
        self.model = model
        self.postprocess = PostProcess(topk=int(topk))

    @torch.inference_mode()
    def predict(
        self,
        feats: Sequence[torch.Tensor],
        orig_size: Optional[Any] = None,
        image_id: Optional[int] = None,
        valid_roi: Optional[dict] = None,
    ):
        if len(feats) != 4:
            raise ValueError("Dinov3PlainDETRHead.predict expects 4 tap feature maps")
        xs = [f.float() for f in feats]
        concat_feat = torch.cat(xs, dim=1)
        device = concat_feat.device
        _, _, feat_h, feat_w = concat_feat.shape
        mask = torch.zeros(concat_feat.size(0), feat_h, feat_w, dtype=torch.bool, device=device)
        feat_nested = NestedTensor(concat_feat, mask)
        pos = [self.model.position_embedding(feat_nested).to(concat_feat.dtype)]
        srcs = [self.model.input_proj[0](concat_feat)]
        masks = [mask]

        det = self.model
        save_nq = det.num_queries
        save_ts = det.transformer.two_stage_num_proposals
        det.num_queries = det.num_queries_one2one
        det.transformer.two_stage_num_proposals = det.num_queries
        try:
            query_embeds = None
            if not det.two_stage or det.mixed_selection:
                query_embeds = det.query_embed.weight[: det.num_queries, :]

            sa_mask = torch.zeros(
                det.num_queries, det.num_queries, dtype=torch.bool, device=device
            )
            sa_mask[det.num_queries_one2one :, : det.num_queries_one2one] = True
            sa_mask[: det.num_queries_one2one, det.num_queries_one2one :] = True

            (
                hs,
                init_ref,
                inter_refs,
                enc_cls,
                enc_coord_unact,
                _enc_delta,
                _output_proposals,
                _max_shape,
            ) = det.transformer(srcs, masks, pos, query_embeds, sa_mask)

            outputs_classes, outputs_coords = [], []
            for lvl in range(hs.shape[0]):
                ref = init_ref if lvl == 0 else inter_refs[lvl - 1]
                ref = inverse_sigmoid(ref)
                oc = det.class_embed[lvl](hs[lvl])
                tmp = det.bbox_embed[lvl](hs[lvl])
                if ref.shape[-1] == 4:
                    tmp = tmp + ref
                else:
                    tmp = tmp.clone()
                    tmp[..., :2] += ref
                outputs_classes.append(oc)
                outputs_coords.append(tmp.sigmoid())

            out = {
                "pred_logits": outputs_classes[-1][:, : det.num_queries_one2one],
                "pred_boxes": outputs_coords[-1][:, : det.num_queries_one2one],
            }
            if det.two_stage:
                out["enc_outputs"] = {
                    "pred_logits": enc_cls,
                    "pred_boxes": enc_coord_unact,
                }

            if orig_size is None:
                orig_h = int(feat_h * 16)
                orig_w = int(feat_w * 16)
            else:
                orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
            canvas_h, canvas_w = int(feat_h * 16), int(feat_w * 16)
            has_padding = valid_roi is not None and (
                valid_roi["top"] != 0
                or valid_roi["left"] != 0
                or valid_roi["height"] != canvas_h
                or valid_roi["width"] != canvas_w
            )
            target_size = (canvas_h, canvas_w) if has_padding else (orig_h, orig_w)
            orig_sizes = torch.tensor([target_size], device=device)
            results = self.postprocess(out, orig_sizes)
            if has_padding:
                offset = torch.tensor(
                    [valid_roi["left"], valid_roi["top"]] * 2,
                    device=device,
                )
                scale = torch.tensor(
                    [orig_w / valid_roi["width"], orig_h / valid_roi["height"]] * 2,
                    device=device,
                )
                for result in results:
                    result["boxes"] = (result["boxes"] - offset) * scale
        finally:
            det.num_queries = save_nq
            det.transformer.two_stage_num_proposals = save_ts

        dets = []
        img_id = int(image_id) if image_id is not None else 0
        for r in results:
            for score, label, box in zip(r["scores"].cpu(), r["labels"].cpu(), r["boxes"].cpu()):
                s = float(score.item())
                if s < self.score_thr:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                dets.append(
                    {
                        "image_id": img_id,
                        "category_id": int(label.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": round(s, 4),
                    }
                )
        return dets
