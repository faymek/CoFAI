from types import SimpleNamespace

import torch
import torch.nn as nn

from cofai.backbone import (
    Dinov3FrozenTail,
    FrozenTail,
)
from cofai.backbone.timm import Dinov3TimmBackbone


def _identity_linear(features: int, scale: float = 1.0) -> nn.Linear:
    layer = nn.Linear(features, features, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(features) * scale)
    return layer


def test_frozen_tail_keeps_input_gradients_and_freezes_parameters():
    tail = FrozenTail(
        [_identity_linear(3)],
        _identity_linear(3, scale=2.0),
    )
    x = torch.randn(2, 4, 3, requires_grad=True)

    output = tail(x)
    output.sum().backward()

    torch.testing.assert_close(output, x.detach() * 2.0)
    assert x.grad is not None
    assert all(not parameter.requires_grad for parameter in tail.parameters())
    assert all(parameter.grad is None for parameter in tail.parameters())

    tail.train()
    assert not tail.training
    assert all(not module.training for module in tail.modules())
    assert not tail.forward_nograd(x).requires_grad


class _Dinov3Block(nn.Module):
    def __init__(self, expected_rope, expected_mask):
        super().__init__()
        self.expected_rope = expected_rope
        self.expected_mask = expected_mask

    def forward(self, x, rope, attn_mask):
        assert rope is self.expected_rope
        assert attn_mask is self.expected_mask
        return x + rope


def test_dinov3_frozen_tail_uses_global_rope_indices():
    ropes = [torch.tensor(float(index)) for index in range(3)]
    attn_mask = torch.ones(1)
    model = SimpleNamespace(
        blocks=nn.ModuleList(
            [
                _Dinov3Block(ropes[0], attn_mask),
                _Dinov3Block(ropes[1], attn_mask),
                _Dinov3Block(ropes[2], attn_mask),
            ]
        ),
        norm=nn.Identity(),
        rope_mixed=True,
    )
    tail = Dinov3FrozenTail(
        model,
        split_layer_idx=0,
        rope=ropes,
        attn_mask=attn_mask,
    )

    output = tail(torch.zeros(1, 2, 3))

    torch.testing.assert_close(output, torch.full((1, 2, 3), 3.0))
    assert len(tail.blocks) == 2


class _Dinov3Model(nn.Module):
    def __init__(self, ropes, attn_mask):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _Dinov3Block(ropes[0], attn_mask),
                _Dinov3Block(ropes[1], attn_mask),
            ]
        )
        self.norm = nn.Identity()
        self.rope_mixed = True


class _Dinov3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.ropes = [torch.tensor(1.0), torch.tensor(2.0)]
        self.attn_mask = torch.ones(1)
        self.model = _Dinov3Model(self.ropes, self.attn_mask)
        self.patch_size = 2
        self._rope = None
        self._attn_mask = None
        self.encoded_image_shape = None

    def encode(self, image):
        self.encoded_image_shape = tuple(image.shape)
        self._rope = self.ropes
        self._attn_mask = self.attn_mask
        return torch.zeros(1, 1, 1)


def test_dinov3_backbone_builds_tail_with_spatial_rope():
    backbone = _Dinov3Backbone()

    tail = Dinov3TimmBackbone.build_frozen_tail(
        backbone,
        layer_idx=0,
        token_hw=(3, 4),
        device="cpu",
    )

    output = tail(torch.zeros(1, 2, 3))
    torch.testing.assert_close(output, torch.full((1, 2, 3), 2.0))
    assert backbone.encoded_image_shape == (1, 3, 6, 8)
