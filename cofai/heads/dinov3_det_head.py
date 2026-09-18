"""Plain-DETR detection head: loads DETR weights only; timm supplies 4-scale taps."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


class _StubDinoViT(nn.Module):
    """Placeholder so ``build_model`` can size DETR without the official ViT."""

    n_blocks = 24
    embed_dim = 1024
    patch_size = 16

    def named_parameters(self, recurse: bool = True):
        return iter(())


class Dinov3PlainDETRHead(nn.Module):
    """Wrap Plain-DETR transformer / box head; features come from CTC timm taps."""

    def __init__(
        self,
        checkpoint: str,
        dinov3_src: str,
        score_thr: float = 0.001,
        topk: int = 100,
    ):
        super().__init__()
        self.score_thr = float(score_thr)
        src = os.path.expanduser(str(dinov3_src))
        if src not in sys.path:
            sys.path.insert(0, src)

        from dinov3.eval.detection.config import DetectionHeadConfig
        from dinov3.eval.detection.models.detr import PostProcess, build_model
        from dinov3.eval.detection.models.position_encoding import PositionEncoding
        from dinov3.eval.detection.util.misc import NestedTensor, inverse_sigmoid

        self._NestedTensor = NestedTensor
        self._inverse_sigmoid = inverse_sigmoid

        backbone_model = _StubDinoViT()
        layers_to_use = [5, 11, 17, 23]
        patch_size = backbone_model.patch_size
        config = DetectionHeadConfig(
            num_classes=91,
            with_box_refine=True,
            two_stage=True,
            mixed_selection=True,
            look_forward_twice=True,
            k_one2many=6,
            lambda_one2many=1.0,
            num_queries_one2one=300,
            num_queries_one2many=300,
            reparam=False,
            topk=int(topk),
            position_embedding=PositionEncoding.SINE,
            num_feature_levels=1,
            dec_layers=6,
            dim_feedforward=2048,
            hidden_dim=768,
            dropout=0.0,
            nheads=8,
            norm_type="pre_norm",
            aux_loss=True,
            proposal_feature_levels=4,
            proposal_min_size=50,
            decoder_type="global_rpe_decomp",
            decoder_use_checkpoint=False,
            decoder_rpe_hidden_dim=512,
            decoder_rpe_type="linear",
            add_transformer_encoder=True,
            num_encoder_layers=6,
            layers_to_use=layers_to_use,
            blocks_to_train=None,
            n_windows_sqrt=0,
            proposal_in_stride=patch_size,
            proposal_tgt_strides=[int(m * patch_size) for m in (0.5, 1, 2, 4)],
            backbone_use_layernorm=False,
        )
        model = build_model(backbone_model, config)
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
        model.load_state_dict(filtered, strict=False)
        model.eval()
        self.model = model
        self.postprocess = PostProcess(topk=int(topk), reparam=False)

    @torch.inference_mode()
    def predict(
        self,
        feats: Sequence[torch.Tensor],
        orig_size: Optional[Any] = None,
        image_id: Optional[int] = None,
    ):
        if not feats:
            raise ValueError("Dinov3PlainDETRHead.predict expects 4 tap feature maps")
        xs = [f.float() for f in feats]
        concat_feat = torch.cat(xs, dim=1)
        device = concat_feat.device
        _, _, feat_h, feat_w = concat_feat.shape
        mask = torch.zeros(concat_feat.size(0), feat_h, feat_w, dtype=torch.bool, device=device)
        feat_nested = self._NestedTensor(concat_feat, mask)
        position_embedding = self.model.backbone[1]
        pos = [position_embedding[0](feat_nested).to(concat_feat.dtype)]
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
                ref = self._inverse_sigmoid(ref)
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
            orig_sizes = torch.tensor([[orig_h, orig_w]], device=device)
            results = self.postprocess(out, orig_sizes)
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
