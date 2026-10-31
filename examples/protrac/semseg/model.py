import torch
from torch import nn
from torch.nn import functional as F

from cofai.backbone import Dinov3TimmBackbone
from examples.protrac.calibration import (
    ProTRACStats,
    clip_delta,
    projection_delta,
    semantic_weights,
)

from .stats import load_stats


class ProTRACSegBackbone(nn.Module):
    """Slot-23 patch calibration and final channel-moment alignment."""

    def __init__(self, cfg, head, device):
        super().__init__()
        self.dino = Dinov3TimmBackbone(
            model_size="large",
            img_size=512,
            dynamic_size=True,
            slot=cfg.slot,
            n_last_blocks=1,
            device=str(device),
            cast_dtype="float16",
            pretrained=False,
            ckpt_path=str(cfg.backbone),
        ).to(device)
        self.head = head
        self.cfg = cfg.calibration
        self.global_source = load_stats(cfg.stats.global_stats, device)
        self.class_source = load_stats(cfg.stats.class_stats, device)
        self.region_source = load_stats(cfg.stats.region_stats, device)
        self.global_stats = ProTRACStats.create(self.global_source["source_means"])
        self.class_stats = ProTRACStats.create(self.class_source["class_means"], self.cfg.class_prior)
        self.region_stats = ProTRACStats.create(self.region_source["source_means"], self.cfg.region_prior)
        self.reset()

    def reset(self):
        self.global_stats.reset()
        self.class_stats.reset(self.cfg.class_prior)
        self.region_stats.reset(self.cfg.region_prior)
        self.moment_mean = self.head.bn.running_mean.detach().float().clone()
        self.moment_second = self.head.bn.running_var.detach().float() + self.moment_mean.square()
        self.moment_count = self.moment_mean.new_tensor(self.cfg.moment_prior)
        self.pending = None

    def patches(self, tokens):
        return tokens[:, self.dino.model.num_prefix_tokens :]

    def feature_map(self, tokens, hw):
        patches = self.patches(tokens)
        return patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[2], *hw)

    def class_assignments(self, logits, patches):
        cfg = self.cfg
        probability = logits.flatten(2).transpose(1, 2).float().softmax(dim=-1)
        confidence = probability.max(dim=-1).values
        values, indices = probability.topk(cfg.class_topk, dim=-1)
        score = values.clamp_min(1e-6).log()
        prototypes = self.class_source["class_prototypes"][indices]
        similarity = torch.einsum("bnd,bnkd->bnk", F.normalize(patches, dim=-1), prototypes)
        weights = (score + cfg.prototype_weight * similarity / cfg.prototype_temperature).softmax(dim=-1)
        gate = ((confidence - cfg.min_confidence) / (1 - cfg.min_confidence)).clamp(0, 1)
        sorted_weights = weights.sort(dim=-1, descending=True).values
        gate = gate * (sorted_weights[..., 0] - sorted_weights[..., 1]).clamp(0, 1)
        return indices, weights, confidence, gate

    def class_shift(self, indices, weights):
        shift = self.class_stats.mean.new_zeros((*indices.shape[:2], self.class_stats.mean.shape[-1]))
        mass = torch.zeros(indices.shape[:2], device=shift.device)
        for candidate in range(indices.shape[-1]):
            index = indices[..., candidate]
            weight = weights[..., candidate] * self.class_source["class_valid"][index]
            delta = self.class_stats.source[index] - self.class_stats.mean[index]
            shift.add_(weight.unsqueeze(-1) * delta)
            mass.add_(weight)
        return shift / mass.clamp_min(1e-6).unsqueeze(-1)

    def region_shift(self, patches):
        cfg = self.cfg
        assignments = semantic_weights(
            patches,
            self.region_source["prototypes"],
            cfg.region_topm,
            cfg.region_temperature,
        )
        translation = clip_delta(cfg.region_scale * self.region_stats.shift(assignments), patches, cfg.region_clip)
        weights, indices = assignments.topk(cfg.region_topm, dim=-1, sorted=False)
        translated = patches + translation
        residual = projection_delta(
            translated.unsqueeze(2), self.region_source["source_means"][indices], self.region_source["bases"][indices]
        )
        quantiles = self.region_source["residual_quantiles"][indices]
        gate = (
            (residual.norm(dim=-1) - quantiles[..., 0]) / (quantiles[..., 2] - quantiles[..., 0]).clamp_min(1e-6)
        ).clamp(0, 1)
        projection = ((weights * gate).unsqueeze(-1) * residual).sum(dim=2)
        return translation + clip_delta(cfg.projection_scale * projection, translated, cfg.projection_clip)

    @torch.no_grad()
    def encode(self, image, valid_roi=None, **kwargs):
        cfg = self.cfg
        hw = (image.shape[-2] // 16, image.shape[-1] // 16)
        tokens = self.dino.encode(image)
        raw_final = self.dino.decode_whole(tokens, hw)[-1]
        raw_logits = self.head([self.feature_map(raw_final, hw)])
        raw_patches = self.patches(tokens).float()
        valid = torch.ones((1, 1, *image.shape[-2:]), device=image.device)
        if valid_roi is not None:
            valid.zero_()
            valid[:, :, valid_roi[0], valid_roi[1]] = 1
        valid = F.avg_pool2d(valid, 16, 16).flatten(1)
        pooled = (raw_patches * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        with torch.autocast(device_type=image.device.type, enabled=False):
            assignments = semantic_weights(
                raw_final[:, 0], self.global_source["semantic_prototypes"], cfg.global_topm, cfg.global_temperature
            )
            global_shift = clip_delta(cfg.global_scale * self.global_stats.shift(assignments), pooled, cfg.global_clip)
            aligned = raw_patches + global_shift.unsqueeze(1)
            indices, weights, confidence, gate = self.class_assignments(raw_logits, aligned)
            class_shift = clip_delta(
                cfg.class_scale * (gate * valid).unsqueeze(-1) * self.class_shift(indices, weights),
                raw_patches,
                cfg.class_clip,
            )
            delta = global_shift.unsqueeze(1) + class_shift + self.region_shift(aligned)
        edited = tokens.clone()
        edited[:, self.dino.model.num_prefix_tokens :] = (raw_patches + delta).to(tokens.dtype)
        final = self.dino.decode_whole(edited, hw)[-1]
        patches = self.patches(final).float()
        with torch.autocast(device_type=image.device.type, enabled=False):
            variance = (self.moment_second - self.moment_mean.square()).clamp_min(1e-4)
            normalized = (patches - self.moment_mean) / variance.sqrt()
            normalized = normalized * self.head.bn.running_var.float().clamp_min(1e-6).sqrt()
            normalized = normalized + self.head.bn.running_mean.float()
            moment_shift = clip_delta(cfg.moment_scale * (normalized - patches), patches, cfg.moment_clip)
        calibrated = final.clone()
        calibrated[:, self.dino.model.num_prefix_tokens :] = (patches + moment_shift).to(final.dtype)
        self.pending = (
            pooled,
            assignments,
            aligned,
            indices,
            weights,
            confidence,
            self.patches(raw_final).float(),
            valid,
        )
        return {"h": calibrated, "pstate": {"patch_hw": hw}}

    def decode(self, h, *, pstate, tasks, **kwargs):
        return {"semseg": [self.feature_map(h, pstate["patch_hw"])]} if "semseg" in tasks else {}

    @torch.no_grad()
    def update(self):
        pooled, assignments, aligned, indices, weights, confidence, raw_final, valid = self.pending
        cfg = self.cfg
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            self.global_stats.update(pooled, assignments)
            region_weights = semantic_weights(
                aligned, self.region_source["prototypes"], cfg.region_topm, cfg.region_temperature
            )
            self.region_stats.update(
                aligned[0], region_weights[0] * valid[0, :, None], min_mass=cfg.min_update_mass, per_image=True
            )
            update_gate = (confidence >= cfg.min_confidence).float() * valid
            mass = self.class_stats.count.new_zeros(self.class_stats.count.shape)
            total = torch.zeros_like(self.class_stats.mean)
            for candidate in range(indices.shape[-1]):
                index = indices[..., candidate].reshape(-1)
                weight = (weights[..., candidate] * update_gate).reshape(-1)
                mass.index_add_(0, index, weight)
                total.index_add_(0, index, aligned.reshape(-1, aligned.shape[-1]) * weight[:, None])
            selected = (mass >= cfg.min_update_mass) & self.class_source["class_valid"].bool()
            count = self.class_stats.count[selected]
            mean = total[selected] / mass[selected, None].clamp_min(1e-6)
            self.class_stats.mean[selected] = (self.class_stats.mean[selected] * count[:, None] + mean) / (
                count[:, None] + 1
            )
            self.class_stats.count[selected] = count + 1
            denominator = valid.sum(dim=1, keepdim=True).clamp_min(1e-6)
            image_mean = (raw_final * valid.unsqueeze(-1)).sum(dim=1) / denominator
            image_second = (raw_final.square() * valid.unsqueeze(-1)).sum(dim=1) / denominator
            count = self.moment_count + 1
            self.moment_mean = (self.moment_mean * self.moment_count + image_mean[0]) / count
            self.moment_second = (self.moment_second * self.moment_count + image_second[0]) / count
            self.moment_count = count
        self.pending = None
