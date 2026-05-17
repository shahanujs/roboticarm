"""
Neural network building blocks used by both Diffusion Policy and ACT.

Key components
--------------
- ResNet18Encoder      : Visual encoder (pretrained ResNet-18, strips classifier)
- FiLMBlock            : Feature-wise Linear Modulation for conditioning
- TransformerEncoder   : Lightweight multi-head self-attention stack
- ConditionalUNet1D    : 1-D U-Net for diffusion denoising
- SinusoidalPosEmb     : Timestep embedding for diffusion
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


# ── Sinusoidal timestep embedding ─────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Maps diffusion timestep t (scalar) → embedding vector of dim `dim`."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int or float
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args   = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)      # (B, dim)


# ── Visual encoder ─────────────────────────────────────────────────────────────

class ResNet18Encoder(nn.Module):
    """
    Pretrained ResNet-18 backbone → pooled feature vector of size 512.
    Input : (B, 3, H, W)  float32, normalised to [0,1]
    Output: (B, 512)
    """

    def __init__(self, pretrained: bool = True, freeze_bn: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base    = models.resnet18(weights=weights)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)

        # Normalisation (ImageNet stats)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.backbone(x).squeeze(-1).squeeze(-1)   # (B, 512)


# ── FiLM conditioning ─────────────────────────────────────────────────────────

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: scale + shift from a conditioning vector."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.fc(cond)               # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # x: (B, C, T)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


# ── 1-D Residual block for U-Net ─────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cond_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels,  out_channels, kernel_size, padding=pad),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
        self.film   = FiLMBlock(out_channels, cond_dim)
        self.skip   = (nn.Conv1d(in_channels, out_channels, 1)
                       if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.film(h, cond)
        h = self.block2(h)
        return h + self.skip(x)


# ── Conditional U-Net 1D ─────────────────────────────────────────────────────

class ConditionalUNet1D(nn.Module):
    """
    1-D U-Net for denoising action sequences.

    Input
    -----
    noisy_action : (B, action_dim, pred_horizon)
    timestep     : (B,)   int diffusion timestep
    global_cond  : (B, cond_dim)   visual + proprioceptive features

    Output
    ------
    noise_pred   : (B, action_dim, pred_horizon)
    """

    def __init__(
        self,
        action_dim:    int   = 7,
        pred_horizon:  int   = 16,
        cond_dim:      int   = 1024,
        diffusion_step_emb_dim: int = 256,
        down_dims:     Tuple[int, ...] = (256, 512, 1024),
        kernel_size:   int   = 5,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        all_cond_dim      = cond_dim + diffusion_step_emb_dim

        # Timestep embedding
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_emb_dim),
            nn.Linear(diffusion_step_emb_dim, diffusion_step_emb_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_emb_dim * 4, diffusion_step_emb_dim),
        )

        # Encoder (down-sampling)
        self.downs: nn.ModuleList = nn.ModuleList()
        self.pools: nn.ModuleList = nn.ModuleList()
        in_ch = action_dim
        for out_ch in down_dims:
            self.downs.append(ResidualBlock1D(in_ch, out_ch, all_cond_dim, kernel_size))
            self.pools.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        # Bottleneck
        self.mid1 = ResidualBlock1D(in_ch, in_ch, all_cond_dim, kernel_size)
        self.mid2 = ResidualBlock1D(in_ch, in_ch, all_cond_dim, kernel_size)

        # Decoder (up-sampling)
        self.ups:     nn.ModuleList = nn.ModuleList()
        self.upconvs: nn.ModuleList = nn.ModuleList()
        for out_ch in reversed(down_dims):
            self.upconvs.append(nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2))
            self.ups.append(ResidualBlock1D(out_ch * 2, out_ch, all_cond_dim, kernel_size))
            in_ch = out_ch

        self.final_conv = nn.Conv1d(in_ch, action_dim, 1)

    def forward(
        self,
        noisy_action: torch.Tensor,   # (B, action_dim, T)
        timestep:     torch.Tensor,   # (B,)
        global_cond:  torch.Tensor,   # (B, cond_dim)
    ) -> torch.Tensor:

        t_emb  = self.t_emb(timestep)                     # (B, step_emb_dim)
        cond   = torch.cat([global_cond, t_emb], dim=-1)  # (B, all_cond_dim)

        # Encode
        x      = noisy_action
        skips  = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x, cond)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        # Decode
        for up, upconv, skip in zip(self.ups, self.upconvs, reversed(skips)):
            x    = upconv(x)
            # Align lengths (may differ by 1 due to stride=2)
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            x = torch.cat([x, skip], dim=1)
            x = up(x, cond)

        return self.final_conv(x)


# ── Lightweight Transformer for ACT ──────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """Standard multi-head self-attention stack."""

    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.norm(self.encoder(src, src_key_padding_mask=mask))


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.norm(self.decoder(tgt, memory))
