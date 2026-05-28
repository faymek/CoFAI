"""
Backbone implementations using transformers library.

Transformers provides many pretrained models that are ready to use with a single line of code. 
Call from_pretrained() to download and load a model’s weights and configuration stored on the Hugging Face Hub.

NOTE: Transformers library does not support dynamic image size, so the img_size parameter is not used.

1. DINOv2 Models:
   - 'facebook/dinov2-small' (ViT-S/14)
   - 'facebook/dinov2-base' (ViT-B/14, default)
   - 'facebook/dinov2-large' (ViT-L/14)
   - 'facebook/dinov2-giant' (ViT-g/14)
   - 'facebook/dinov2-small-with-registers' (with register tokens)
   - 'facebook/dinov2-base-with-registers' (with register tokens)
   - 'facebook/dinov2-large-with-registers' (with register tokens)
   - 'facebook/dinov2-giant-with-registers' (with register tokens)

2. MAE Models:
   - 'facebook/vit-mae-base' (default, ViT-B/16)
   - 'facebook/vit-mae-large' (ViT-L/16)
   - 'facebook/vit-mae-huge' (ViT-H/16)

3. SigLIP2 Models:
   - 'google/siglip2-base-patch16-256' (default, 256x256 input)
   - 'google/siglip2-base-patch16-384' (384x384 input)
   - 'google/siglip2-base-patch16-512' (512x512 input)
   - 'google/siglip2-large-patch16-256' (large variant, 256x256)
   - 'google/siglip2-large-patch16-384' (large variant, 384x384)
   - 'google/siglip2-large-patch16-512' (large variant, 512x512)
"""

import torch
import torch.nn as nn
from typing import Protocol
from transformers import ViTMAEForPreTraining, SiglipModel, AutoImageProcessor, Dinov2Model, Dinov2WithRegistersModel
from torchvision import transforms


class BackboneProtocol(Protocol):
    """Protocol defining the interface for backbone models."""
    
    patch_size: int
    hidden_size: int
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to intermediate features.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Encoded features of shape (B, N, C)
        """
        ...
    
    def decode(self, h: torch.Tensor, tasks: list[str]) -> dict[str, torch.Tensor]:
        """Decode encoded features for downstream tasks (task set is backbone-specific)."""
        ...


class MAETransformersBackbone(nn.Module):
    """
    MAE (Masked Autoencoder) backbone using transformers library.
    
    This backbone uses ViTMAE from transformers library and provides
    encode/decode functionality for RAE.
    
    Args:
        model_name (str): HuggingFace model name, e.g., 'facebook/vit-mae-base'.
                         Defaults to 'facebook/vit-mae-base'.
        img_size (int): Input image size. Defaults to 256.
        slot (int): Block slicing position for feature extraction. 
                   -1 means use all blocks. Defaults to -1.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        img_size: int = 256,
        slot: int = -1,  # -1 means use all blocks
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.slot = slot
        
        # Load MAE model
        mae_model = ViTMAEForPreTraining.from_pretrained(model_name)
        self.model = mae_model.vit
        
        # Remove the affine of final layernorm (for RAE compatibility)
        self.model.layernorm.elementwise_affine = False
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        
        # Set mask_ratio to 0 (no masking)
        self.model.config.mask_ratio = 0.0
        
        # Get model properties
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        
        # Get image processor for normalization
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.processor.image_mean,
                    std=self.processor.image_std
                ),
            ]
        )
        
        self.model.eval()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images through the MAE encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
        
        Returns:
            h (torch.Tensor): Encoded features, shape (B, N, C).
        """
        # Apply normalization
        x = self.input_transform(x)
        
        # Calculate number of patches
        B, C, H, W = x.shape
        patch_num = int(H * W // (self.patch_size ** 2))
        assert patch_num * self.patch_size ** 2 == H * W, (
            f'image size should be divisible by patch size: {H}x{W} vs patch_size={self.patch_size}'
        )
        
        # Create noise (position indices) for MAE
        noise = torch.arange(patch_num, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)
        
        # For MAE, we process through all blocks in encode
        # The slot mechanism is not directly applicable to MAE's structure
        # We'll use all blocks and extract features after slot in decode_rae
        outputs = self.model(x, noise, interpolate_pos_encoding=True)
        h = outputs.last_hidden_state
        
        return h


class SigLIP2TransformersBackbone(nn.Module):
    """
    SigLIP2 backbone using transformers library.
    
    This backbone uses SiglipModel from transformers library and provides
    encode/decode functionality for RAE.
    
    Args:
        model_name (str): HuggingFace model name, e.g., 'google/siglip2-base-patch16-256'.
                         Defaults to 'google/siglip2-base-patch16-256'.
        img_size (int): Input image size. Defaults to 256.
        slot (int): Block slicing position for feature extraction. 
                   -1 means use all blocks. Defaults to -1.
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-256",
        img_size: int = 256,
        slot: int = -1,  # -1 means use all blocks
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.slot = slot
        
        # Load SigLIP2 model
        siglip_model = SiglipModel.from_pretrained(model_name)
        self.model = siglip_model.vision_model
        
        # Remove the affine of final layernorm (for RAE compatibility)
        self.model.post_layernorm.elementwise_affine = False
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None
        
        # Get model properties
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        
        # Get image processor for normalization
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.processor.image_mean,
                    std=self.processor.image_std
                ),
            ]
        )
        
        self.model.eval()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images through the SigLIP2 encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
        
        Returns:
            h (torch.Tensor): Encoded features, shape (B, N, C).
        """
        # Apply normalization
        x = self.input_transform(x)
        
        # Process through all blocks
        outputs = self.model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        h = outputs.last_hidden_state
        
        return h


class Dinov2TransformersBackbone(nn.Module):
    """
    DINOv2 backbone using transformers library.
    
    This backbone uses Dinov2Model or Dinov2WithRegistersModel from transformers library 
    and provides encode/decode functionality for RAE.
    
    Args:
        model_name (str): HuggingFace model name, e.g., 'facebook/dinov2-base' or 
                         'facebook/dinov2-with-registers-base'.
                         Defaults to 'facebook/dinov2-base'.
        img_size (int): Input image size. Defaults to 224.
        slot (int): Block slicing position for feature extraction. 
                   -4 means the last 4th block. -1 means use all blocks (matches old behavior).
                   Defaults to -4.
        normalize (bool): Whether to remove layernorm affine parameters (for RAE compatibility).
                         Defaults to True.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        img_size: int = 224,
        slot: int = -4,
        normalize: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.slot = slot
        
        # Determine if model has registers based on model_name
        has_registers = "with-registers" in model_name.lower() or "with_registers" in model_name.lower()
        
        # Load DINOv2 model (with or without registers)
        try:
            if has_registers:
                self.model = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=True)
            else:
                self.model = Dinov2Model.from_pretrained(model_name, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            # Fallback to downloading from HuggingFace
            if has_registers:
                self.model = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=False)
            else:
                self.model = Dinov2Model.from_pretrained(model_name, local_files_only=False)
        
        # Remove layernorm affine parameters if normalize=True (for RAE compatibility)
        if normalize:
            if hasattr(self.model, 'layernorm'):
                self.model.layernorm.elementwise_affine = False
                self.model.layernorm.weight = None
                self.model.layernorm.bias = None
        
        # Get model properties
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        
        # Get image processor for normalization
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.processor.image_mean,
                    std=self.processor.image_std
                ),
            ]
        )
        
        self.model.eval()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images through the DINOv2 encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
        
        Returns:
            h (torch.Tensor): Encoded features, shape (B, N, C).
        """
        # Apply normalization
        x = self.input_transform(x)
        
        # Get embeddings
        embeddings = self.model.embeddings(x)
        
        # Process through blocks[:slot]
        # If slot=-1, use all blocks (matches old Dinov2withNorm behavior)
        total_blocks = len(self.model.encoder.layer)
        if self.slot == -1:
            # Use all blocks
            slot_idx = total_blocks
        elif self.slot < 0:
            slot_idx = total_blocks + self.slot
        else:
            slot_idx = self.slot
        
        hidden_states = embeddings
        for i, layer in enumerate(self.model.encoder.layer[:slot_idx]):
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        h = hidden_states
        
        return h


