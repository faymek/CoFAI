import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleLoss(nn.Module):
    """Simple loss function that extracts loss directly from model output.

    This loss function is a wrapper that simply retrieves the loss value
    from the model output dictionary and returns it along with monitoring metrics.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize SimpleLoss.

        Args:
            **kwargs (dict): Additional keyword arguments (currently unused).
        """
        super().__init__()

    def forward(self, output, x):
        """Forward pass to compute loss.

        Args:
            output (dict): Model output dictionary containing:

                - "loss" (torch.Tensor): Pre-computed loss tensor
                - "monitor" (dict, optional): Additional monitoring metrics
            x (torch.Tensor): Input tensor (not used, kept for interface compatibility)

        Returns:
            loss (torch.Tensor): The loss tensor
            monitor (dict): Dictionary of monitoring metrics including loss value and additional metrics from output["monitor"] if present
        """
        loss = output["loss"]
        monitor = {
            "loss": loss.detach().mean().item(),
        }
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


class MPC2Loss(nn.Module):
    """MPC2 loss function combining rate-distortion optimization.

    This loss function combines:

    - Bits per pixel (BPP) loss from likelihoods
    - PATCH tokens reconstruction loss
    - CLS token reconstruction loss

    The total loss is: rlmbda * bpp_loss + cls_token_loss + patch_tokens_loss
    """

    def __init__(
        self,
        rlmbda=1.0,
    ):
        """Initialize MPC2Loss.

        Args:
            rlmbda (float, optional): Rate-distortion trade-off parameter.
                Higher values emphasize compression rate. Defaults to 1.0.
        """
        super().__init__()
        self.rlmbda = rlmbda

    def get_rlmbda(self, global_step=None):
        """Get the rate-distortion trade-off parameter.

        Args:
            global_step (int, optional): Current training step (currently unused).
                Can be used for scheduled lambda values in the future.

        Returns:
            rlmbda (float): The rate-distortion trade-off parameter.
        """
        return self.rlmbda

    def forward(self, output, x, x_shape=None, global_step=None):
        """Forward pass to compute MPC2 loss.

        Args:
            output (dict): Model output dictionary containing:

                - "likelihoods" (dict): Dictionary of likelihood tensors for BPP calculation
                - "h_dino_hat" (torch.Tensor): Reconstructed DINO features [B, N+1, D]
                - "h_dino" (torch.Tensor): Target DINO features [B, N+1, D]
                - "monitor" (dict, optional): Additional monitoring metrics
            x (torch.Tensor, optional): Input tensor [B, C, H, W].
                Used to infer shape if x_shape is None.
            x_shape (tuple, optional): Shape of input tensor (N, C, H, W).
                Required if x is None.
            global_step (int, optional): Current training step for lambda scheduling.

        Returns:
            loss (torch.Tensor): The total loss tensor
            monitor (dict): Dictionary of monitoring metrics including loss value and additional metrics from output["monitor"] if present

        Raises:
            ValueError: If both x and x_shape are None.
        """
        if x is None and x_shape is None:
            raise ValueError("x and x_shape cannot be both None")
        x_shape = x.shape if x is not None else x_shape
        N, _, H, W = x_shape
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_dino_hat = output["h_dino_hat"].contiguous()
        h_dino = output["h_dino"].contiguous()

        h_dino_loss = F.mse_loss(h_dino_hat[:, 1:, :], h_dino[:, 1:, :])
        cls_token_loss = F.mse_loss(h_dino_hat[:, 0, :], h_dino[:, 0, :])

        rlmbda = self.get_rlmbda(global_step)
        loss = rlmbda * bpp_loss + cls_token_loss + h_dino_loss

        monitor = {
            "loss": loss.detach().mean().item(),
            "cls": cls_token_loss.detach().mean().item(),
            "dino": h_dino_loss.detach().mean().item(),
            "bpp": bpp_loss.item(),
            "rlmbda": rlmbda,
        }
        monitor.update(
            {name: bpp.detach().mean().item() for name, bpp in bpp_components.items()}
        )
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


class MPC12Loss(nn.Module):
    """MPC12 loss function combining rate-distortion optimization.

    This loss function combines:

    - Bits per pixel (BPP) loss from likelihoods
    - VQGAN feature reconstruction loss
    - PATCH tokens reconstruction loss
    - CLS token reconstruction loss

    The total loss is: rlmbda * bpp_loss + cls_token_loss + patch_tokens_loss + h_vqgan_loss
    """

    def __init__(
        self,
        rlmbda=1.0,
    ):
        """Initialize MPC12Loss.

        Args:
            rlmbda (float, optional): Rate-distortion trade-off parameter.
                Higher values emphasize compression rate. Defaults to 1.0.
        """
        super().__init__()
        self.rlmbda = rlmbda

    def get_rlmbda(self, global_step=None):
        """Get the rate-distortion trade-off parameter.

        Args:
            global_step (int, optional): Current training step (currently unused).
                Can be used for scheduled lambda values in the future.

        Returns:
            rlmbda (float): The rate-distortion trade-off parameter.
        """
        return self.rlmbda

    def forward(self, output, x, x_shape=None, global_step=None):
        """Forward pass to compute MPC12 loss.

        Args:
            output (dict): Model output dictionary containing:

                - "likelihoods" (dict): Dictionary of likelihood tensors for BPP calculation
                - "h_vqgan_hat" (torch.Tensor): Reconstructed VQGAN features
                - "h_vqgan" (torch.Tensor): Target VQGAN features
                - "h_dino_hat" (torch.Tensor): Reconstructed DINO features [B, N+1, D]
                - "h_dino" (torch.Tensor): Target DINO features [B, N+1, D]
                - "monitor" (dict, optional): Additional monitoring metrics
            x (torch.Tensor, optional): Input tensor [B, C, H, W].
                Used to infer shape if x_shape is None.
            x_shape (tuple, optional): Shape of input tensor (N, C, H, W).
                Required if x is None.
            global_step (int, optional): Current training step for lambda scheduling.

        Returns:
            loss (torch.Tensor): The total loss tensor
            monitor (dict): Dictionary of monitoring metrics including loss value and additional metrics from output["monitor"] if present

        Raises:
            ValueError: If both x and x_shape are None.
        """
        if x is None and x_shape is None:
            raise ValueError("x and x_shape cannot be both None")
        x_shape = x.shape if x is not None else x_shape
        N, _, H, W = x_shape
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_vqgan_hat = output["h_vqgan_hat"].contiguous()
        h_vqgan = output["h_vqgan"].contiguous()

        h_vqgan_loss = F.mse_loss(h_vqgan_hat, h_vqgan)

        h_dino_hat = output["h_dino_hat"].contiguous()
        h_dino = output["h_dino"].contiguous()

        h_dino_loss = F.mse_loss(h_dino_hat[:, 1:, :], h_dino[:, 1:, :])
        cls_token_loss = F.mse_loss(h_dino_hat[:, 0, :], h_dino[:, 0, :])

        rlmbda = self.get_rlmbda(global_step)
        loss = rlmbda * bpp_loss + cls_token_loss + h_dino_loss + h_vqgan_loss

        monitor = {
            "loss": loss.detach().mean().item(),
            "cls": cls_token_loss.detach().mean().item(),
            "dino": h_dino_loss.detach().mean().item(),
            "vqgan": h_vqgan_loss.detach().mean().item(),
            "bpp": bpp_loss.item(),
            "rlmbda": rlmbda,
        }
        monitor.update(
            {name: bpp.detach().mean().item() for name, bpp in bpp_components.items()}
        )
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor

class MPC3Loss(nn.Module):
    """MPC3 loss function combining rate-distortion optimization.

    This loss function combines:

    - Bits per pixel (BPP) loss from likelihoods
    - PATCH tokens reconstruction loss
    - CLS token reconstruction loss

    The total loss is: rlmbda * bpp_loss + cls_token_loss + patch_tokens_loss
    """

    def __init__(
        self,
        rlmbda=1.0,
        num_prefix_tokens=5,
        likelihood_eps=1e-9,
    ):
        """Initialize MPC2Loss.

        Args:
            rlmbda (float, optional): Rate-distortion trade-off parameter.
                Higher values emphasize compression rate. Defaults to 1.0.
        """
        super().__init__()
        self.rlmbda = rlmbda
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.likelihood_eps = float(likelihood_eps)

    def _resolve_num_prefix_tokens(self, output, override=None):
        if override is not None:
            return int(override)
        if "num_prefix_tokens" in output:
            return int(output["num_prefix_tokens"])
        return self.num_prefix_tokens

    def get_rlmbda(self, global_step=None):
        """Get the rate-distortion trade-off parameter.

        Args:
            global_step (int, optional): Current training step (currently unused).
                Can be used for scheduled lambda values in the future.

        Returns:
            rlmbda (float): The rate-distortion trade-off parameter.
        """
        return self.rlmbda

    def forward(
        self,
        output,
        x,
        x_shape=None,
        global_step=None,
        rlmbda=None,
        lmbda=None,
        num_prefix_tokens=None,
    ):
        """Forward pass to compute MPC2 loss.

        Args:
            output (dict): Model output dictionary containing:

                - "likelihoods" (dict): Dictionary of likelihood tensors for BPP calculation
                - "h_dino_hat" (torch.Tensor): Reconstructed DINO features [B, N+1, D]
                - "h_dino" (torch.Tensor): Target DINO features [B, N+1, D]
                - "monitor" (dict, optional): Additional monitoring metrics
            x (torch.Tensor, optional): Input tensor [B, C, H, W].
                Used to infer shape if x_shape is None.
            x_shape (tuple, optional): Shape of input tensor (N, C, H, W).
                Required if x is None.
            global_step (int, optional): Current training step for lambda scheduling.
            rlmbda (float, optional): Rate-distortion trade-off parameter.
        Returns:
            loss (torch.Tensor): The total loss tensor
            monitor (dict): Dictionary of monitoring metrics including loss value and additional metrics from output["monitor"] if present

        Raises:
            ValueError: If both x and x_shape are None.
        """
        if x is None and x_shape is None:
            raise ValueError("x and x_shape cannot be both None")
        x_shape = x.shape if x is not None else x_shape
        N, _, H, W = x_shape
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods.clamp_min(self.likelihood_eps)).sum() / (
                -math.log(2) * num_pixels
            )
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_dino_hat = output["h_dino_hat"].contiguous()
        h_dino = output["h_dino"].contiguous()

        n_prefix = self._resolve_num_prefix_tokens(output, override=num_prefix_tokens)
        if n_prefix < 1:
            raise ValueError(f"num_prefix_tokens must be >= 1, got {n_prefix}")
        if h_dino_hat.shape[1] <= n_prefix:
            raise ValueError(
                f"Token length ({h_dino_hat.shape[1]}) must be > num_prefix_tokens ({n_prefix})."
            )

        h_dino_loss = F.mse_loss(h_dino_hat[:, n_prefix:, :], h_dino[:, n_prefix:, :])
        cls_token_loss = F.mse_loss(h_dino_hat[:, :1, :], h_dino[:, :1, :])

        # 优先使用调用时传入的 lambda（兼容 lmbda / rlmbda）
        if lmbda is not None:
            rlmbda_eff = float(lmbda)
        elif rlmbda is not None:
            rlmbda_eff = float(rlmbda)
        else:
            rlmbda_eff = float(self.get_rlmbda(global_step))

        loss = rlmbda_eff * bpp_loss + cls_token_loss + h_dino_loss

        monitor = {
            "loss": loss.detach().mean().item(),
            "cls": cls_token_loss.detach().mean().item(),
            "dino": h_dino_loss.detach().mean().item(),
            "bpp": bpp_loss.item(),
            "rlmbda": rlmbda_eff,
        }
        monitor.update(
            {name: bpp.detach().mean().item() for name, bpp in bpp_components.items()}
        )
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


