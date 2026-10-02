import json

import numpy as np
import pytest
import torch

from cofai.latent_codecs.vq_ufc import VQUFCFeatureCodec
from cofai.utils.utils_vqufc import load_rms, split_features, split_hash


def make_codec(prefix_rms: torch.Tensor, patch_rms: torch.Tensor) -> VQUFCFeatureCodec:
    codec = VQUFCFeatureCodec.__new__(VQUFCFeatureCodec)
    torch.nn.Module.__init__(codec)
    codec.num_prefix_tokens = prefix_rms.shape[0]
    codec.register_buffer("prefix_position_channel_rms", prefix_rms)
    codec.register_buffer("patch_channel_rms", patch_rms)
    return codec


def test_all_tokens_use_position_and_patch_rms_and_round_trip():
    prefix_rms = torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
    patch_rms = torch.tensor([4.0, 5.0, 10.0])
    codec = make_codec(prefix_rms, patch_rms)
    features = torch.tensor([[[1.0, 4.0, 12.0], [4.0, 12.0, 32.0], [8.0, 15.0, 40.0]]])

    normalized, context = codec._prepare_input(features)

    assert normalized.shape == features.shape
    torch.testing.assert_close(
        normalized,
        torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]]),
    )
    torch.testing.assert_close(codec._restore_output(normalized, context), features)


def test_all_token_mode_requires_patch_tokens():
    codec = make_codec(torch.ones(2, 3), torch.ones(3))
    with pytest.raises(ValueError, match=r"Expected prefix\+patch tokens"):
        codec._prepare_input(torch.ones(1, 2, 3))


def test_combined_rms_is_bound_to_the_training_split(tmp_path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    for index in range(4):
        np.save(feature_root / f"{index}.npy", np.ones((7, 3), dtype=np.float32))
    root, train_files, _ = split_features(feature_root, seed=42, val_ratio=0.25)
    metadata = {
        "split_sha256": split_hash(train_files, root),
        "num_prefix_tokens": 2,
    }
    artifact = tmp_path / "rms.npz"
    np.savez_compressed(
        artifact,
        prefix_position_channel_rms=np.ones((2, 3), dtype=np.float32),
        patch_channel_rms=np.ones(3, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    prefix, patch, loaded = load_rms(artifact, train_files, root, num_prefix_tokens=2)

    assert prefix.shape == (2, 3)
    assert patch.shape == (3,)
    assert loaded["split_sha256"] == metadata["split_sha256"]
