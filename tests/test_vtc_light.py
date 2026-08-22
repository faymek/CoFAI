import torch

from cofai.latent_codecs.vtc import VTCLight
from cofai.layers.locality_aware import LocalityAwareBlock


def test_locality_aware_block_output_shape():
    block = LocalityAwareBlock(64, 64)
    tokens = torch.randn(2, 21, 64)

    output = block(tokens, (4, 4))

    assert output.shape == tokens.shape


def test_vtc_light_state_dict_round_trip():
    model = VTCLight(h_dim=64, y_dim=16, z_dim=4, num_prefix_tokens=1)
    reloaded = VTCLight(h_dim=64, y_dim=16, z_dim=4, num_prefix_tokens=1)

    reloaded.load_state_dict(model.state_dict(), strict=True)

    keys = model.state_dict().keys()
    assert model.y_lc.__class__.__name__ == "StreamlinedEntropyModel"
    assert any(key.startswith("pre_blocks.") for key in keys)
    assert any(key.startswith("post_blocks.") for key in keys)
