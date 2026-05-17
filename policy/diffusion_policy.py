"""
Diffusion Policy — Chi et al., 2023  (https://arxiv.org/abs/2303.04137)

Architecture
------------
Observation encoder:
  - Two ResNet-18 encoders (wrist cam + stand cam)  → 512-D each
  - Proprioceptive MLP                              → 64-D
  - Concatenated → 1088-D global condition

Denoising network:
  - Conditional 1-D U-Net conditioned on global_cond
  - Predicts noise added to action chunk

Inference:
  - DDPM with T=100 noise steps, cosine schedule
  - 'obs_horizon' past observations as condition
  - 'pred_horizon' future actions predicted
  - 'action_horizon' executed before re-planning (temporal ensemble optional)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.networks import ResNet18Encoder, ConditionalUNet1D

logger = logging.getLogger(__name__)

# ── DDPM noise schedule ────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f     = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha = f / f[0]
    betas = 1 - alpha[1:] / alpha[:-1]
    return betas.clamp(0, 0.9999).float()


class DDPMScheduler:
    """Simple DDPM forward / reverse process."""

    def __init__(self, T: int = 100, device: str = "cpu"):
        self.T      = T
        self.device = device

        betas            = cosine_beta_schedule(T).to(device)
        alphas           = 1.0 - betas
        alpha_cumprod    = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        self.register = {}
        def reg(name, val):
            self.register[name] = val

        reg("betas",              betas)
        reg("alphas",             alphas)
        reg("alpha_cumprod",      alpha_cumprod)
        reg("alpha_cumprod_prev", alpha_cumprod_prev)
        reg("sqrt_alpha_cumprod",      alpha_cumprod.sqrt())
        reg("sqrt_one_minus_alpha_cp", (1 - alpha_cumprod).sqrt())
        reg("posterior_variance",
            betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod + 1e-8))

    def __getattr__(self, name):
        if name in ("T", "device", "register"):
            return super().__getattribute__(name)
        return self.register[name]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ap  = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_omap = self.sqrt_one_minus_alpha_cp[t].view(-1, 1, 1)
        return sqrt_ap * x0 + sqrt_omap * noise, noise

    @torch.no_grad()
    def p_sample(self, model_output: torch.Tensor,
                 x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One reverse step: p(x_{t-1} | x_t)."""
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        betas_t     = self.betas[t_tensor].view(-1, 1, 1)
        sqrt_omap_t = self.sqrt_one_minus_alpha_cp[t_tensor].view(-1, 1, 1)
        alphas_t    = self.alphas[t_tensor].view(-1, 1, 1)

        # Predict x_0 from noise
        x0_pred = (x_t - sqrt_omap_t * model_output) / (
            self.sqrt_alpha_cumprod[t_tensor].view(-1, 1, 1) + 1e-8
        )
        x0_pred = x0_pred.clamp(-1, 1)

        # Posterior mean
        coef1   = betas_t * self.alpha_cumprod_prev[t_tensor].view(-1, 1, 1).sqrt() / (
            1 - self.alpha_cumprod[t_tensor].view(-1, 1, 1) + 1e-8
        )
        coef2   = (1 - self.alpha_cumprod_prev[t_tensor].view(-1, 1, 1)) * alphas_t.sqrt() / (
            1 - self.alpha_cumprod[t_tensor].view(-1, 1, 1) + 1e-8
        )
        mean    = coef1 * x0_pred + coef2 * x_t

        if t == 0:
            return mean
        var  = self.posterior_variance[t_tensor].view(-1, 1, 1)
        return mean + var.sqrt() * torch.randn_like(x_t)


# ── Full Diffusion Policy ─────────────────────────────────────────────────────

class DiffusionPolicy(nn.Module):
    """
    Image-based Diffusion Policy for SO-ARM101 orange pick task.

    action_dim   : 7  (6 joints + 1 gripper, normalised to [-1, 1])
    obs_horizon  : 2  (current + 1 past observation)
    pred_horizon : 16 (predict 16 future action steps)
    action_horizon: 8 (execute first 8 before re-planning)
    """

    def __init__(
        self,
        action_dim:    int = 7,
        obs_horizon:   int = 2,
        pred_horizon:  int = 16,
        action_horizon: int = 8,
        diffusion_steps: int = 100,
        device: str = "cuda",
    ):
        super().__init__()
        self.action_dim     = action_dim
        self.obs_horizon    = obs_horizon
        self.pred_horizon   = pred_horizon
        self.action_horizon = action_horizon
        self.device         = device

        # Visual encoders — one per camera
        self.wrist_encoder = ResNet18Encoder(pretrained=True)
        self.stand_encoder = ResNet18Encoder(pretrained=True)

        # Proprioception encoder  (joint angles + gripper)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(action_dim * obs_horizon, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Global condition dim: 512*2 (wrist) + 512*2 (stand) + 64 proprio
        # (multiplied by obs_horizon since we stack observations)
        vis_feat_dim = 512 * 2 * obs_horizon   # wrist+stand per timestep
        cond_dim     = vis_feat_dim + 64

        # 1-D U-Net denoiser
        self.noise_pred_net = ConditionalUNet1D(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            cond_dim=cond_dim,
            diffusion_step_emb_dim=256,
            down_dims=(256, 512, 1024),
        )

        # DDPM scheduler
        self.scheduler = DDPMScheduler(T=diffusion_steps, device=device)

        # Action normalisation statistics (set during training)
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std",  torch.ones(action_dim))

        self.to(device)

    # ── Observation encoding ──────────────────────────────────────────────────

    def encode_obs(
        self,
        wrist_imgs: torch.Tensor,   # (B, obs_horizon, 3, H, W)
        stand_imgs: torch.Tensor,   # (B, obs_horizon, 3, H, W)
        proprio:    torch.Tensor,   # (B, obs_horizon, action_dim)
    ) -> torch.Tensor:
        """Return flattened global condition vector (B, cond_dim)."""
        B, T = wrist_imgs.shape[:2]

        # Flatten time into batch for batch vision encoding
        w_feat = self.wrist_encoder(wrist_imgs.view(B * T, *wrist_imgs.shape[2:]))
        s_feat = self.stand_encoder(stand_imgs.view(B * T, *stand_imgs.shape[2:]))
        # (B*T, 512) → (B, T*512) for each
        w_feat = w_feat.view(B, -1)
        s_feat = s_feat.view(B, -1)

        p_feat = self.proprio_mlp(proprio.view(B, -1))   # (B, 64)

        return torch.cat([w_feat, s_feat, p_feat], dim=-1)  # (B, cond_dim)

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        wrist_imgs: torch.Tensor,
        stand_imgs: torch.Tensor,
        proprio:    torch.Tensor,
        actions:    torch.Tensor,   # (B, pred_horizon, action_dim)
    ) -> torch.Tensor:
        """Returns MSE loss on predicted noise."""
        B = actions.shape[0]

        # Normalise actions
        nactions = (actions - self.action_mean) / (self.action_std + 1e-8)
        nactions = nactions.permute(0, 2, 1)   # (B, action_dim, pred_horizon)

        # Sample random timestep
        t = torch.randint(0, self.scheduler.T, (B,), device=self.device)

        # Add noise
        noisy_actions, noise = self.scheduler.q_sample(nactions, t)

        # Encode observation
        cond = self.encode_obs(wrist_imgs, stand_imgs, proprio)

        # Predict noise
        noise_pred = self.noise_pred_net(noisy_actions, t, cond)

        return F.mse_loss(noise_pred, noise)

    # ── Inference (DDPM reverse) ──────────────────────────────────────────────

    @torch.no_grad()
    def predict_action(
        self,
        wrist_imgs: torch.Tensor,   # (1, obs_horizon, 3, H, W)
        stand_imgs: torch.Tensor,
        proprio:    torch.Tensor,   # (1, obs_horizon, action_dim)
    ) -> np.ndarray:
        """
        Run the full DDPM reverse chain.
        Returns predicted actions (action_horizon, action_dim) in original space.
        """
        self.eval()
        B = 1

        cond = self.encode_obs(wrist_imgs, stand_imgs, proprio)   # (1, cond_dim)

        # Start from pure noise
        x_t = torch.randn(
            B, self.action_dim, self.pred_horizon, device=self.device
        )

        # Reverse diffusion
        for t in reversed(range(self.scheduler.T)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.noise_pred_net(x_t, t_tensor, cond)
            x_t = self.scheduler.p_sample(noise_pred, x_t, t)

        # (B, action_dim, pred_horizon) → (pred_horizon, action_dim)
        x_t = x_t.squeeze(0).permute(1, 0)   # (pred_horizon, action_dim)

        # Denormalise
        actions = x_t * (self.action_std + 1e-8) + self.action_mean

        # Return only the execution horizon
        return actions[:self.action_horizon].cpu().numpy()

    # ── Normalisation stats ───────────────────────────────────────────────────

    def set_normalisation_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.action_mean = torch.from_numpy(mean).float().to(self.device)
        self.action_std  = torch.from_numpy(std).float().to(self.device)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info("DiffusionPolicy saved to %s", path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("DiffusionPolicy loaded from %s", path)
