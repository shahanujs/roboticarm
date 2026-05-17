"""
ACT — Action Chunking with Transformers  (Zhao et al., 2023)
https://arxiv.org/abs/2304.13705

Architecture
------------
Encoder (CVAE encoder, used only at TRAIN time):
  - Encodes target action chunk + proprio → style vector z (mean, log_var)

Policy decoder (used at both train and inference):
  - ResNet-18 vision features for each camera
  - Cross-attention transformer decoder: query=proprio+z, key/value=vision
  - Linear head → action chunk (chunk_size, action_dim)

At INFERENCE:
  - z is sampled from N(0, I) (or zeroed out → deterministic mode)
  - Temporal ensemble: average overlapping predictions for smooth execution
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.networks import (
    ResNet18Encoder, TransformerEncoder, TransformerDecoder
)

logger = logging.getLogger(__name__)


class ACTPolicy(nn.Module):
    """
    Image-conditioned ACT policy for SO-ARM101.

    action_dim  : 7  (6 joints + gripper, in radians / [0,1])
    chunk_size  : 50 (predict 50 actions at once)
    d_model     : 512
    """

    def __init__(
        self,
        action_dim:  int = 7,
        chunk_size:  int = 50,
        d_model:     int = 512,
        nhead:       int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 7,
        z_dim:       int = 32,
        device:      str = "cuda",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.z_dim      = z_dim
        self.device     = device

        # ── Vision encoders ───────────────────────────────────────────────────
        self.wrist_encoder = ResNet18Encoder(pretrained=True)
        self.stand_encoder = ResNet18Encoder(pretrained=True)

        # Project 512-D visual features to d_model
        self.wrist_proj = nn.Linear(512, d_model)
        self.stand_proj = nn.Linear(512, d_model)

        # ── CVAE Encoder (train-time only) ────────────────────────────────────
        # Input: (proprio + action_chunk) tokenised → encode to z
        self.action_enc     = nn.Linear(action_dim, d_model)
        self.proprio_enc    = nn.Linear(action_dim, d_model)
        self.cvae_enc       = TransformerEncoder(d_model=d_model, nhead=nhead,
                                                  num_layers=num_enc_layers)
        self.z_mu_fc        = nn.Linear(d_model, z_dim)
        self.z_logvar_fc    = nn.Linear(d_model, z_dim)

        # ── Policy Transformer decoder ────────────────────────────────────────
        # Queries: positional embeddings over chunk_size
        self.query_embed    = nn.Embedding(chunk_size, d_model)

        # z → d_model
        self.z_proj         = nn.Linear(z_dim, d_model)

        # Proprio → d_model
        self.proprio_dec_fc = nn.Linear(action_dim, d_model)

        # Cross-attention decoder
        self.transformer_dec = TransformerDecoder(d_model=d_model, nhead=nhead,
                                                   num_layers=num_dec_layers)

        # Output head
        self.action_head = nn.Linear(d_model, action_dim)

        # ── Normalisation buffers ─────────────────────────────────────────────
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std",  torch.ones(action_dim))

        # ── Temporal ensemble ─────────────────────────────────────────────────
        self._ensemble_buffer: List[np.ndarray] = []   # sliding window
        self._ensemble_weights = np.exp(
            -0.01 * np.arange(chunk_size)
        )   # decay weight

        self.to(device)

    # ── CVAE encoder (train-time) ─────────────────────────────────────────────

    def encode_style(
        self,
        proprio:    torch.Tensor,   # (B, action_dim)
        actions:    torch.Tensor,   # (B, chunk_size, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var) of latent style z."""
        B = proprio.shape[0]

        # Tokenise: [CLS proprio] [action_0 ... action_{T-1}]
        cls_token = self.proprio_enc(proprio).unsqueeze(1)         # (B,1,d)
        act_tokens = self.action_enc(actions)                      # (B,T,d)
        seq = torch.cat([cls_token, act_tokens], dim=1)            # (B,T+1,d)

        enc_out = self.cvae_enc(seq)    # (B, T+1, d)
        cls_out = enc_out[:, 0, :]     # (B, d)

        return self.z_mu_fc(cls_out), self.z_logvar_fc(cls_out)

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = (0.5 * log_var).exp()
        return mu + std * torch.randn_like(std)

    # ── Policy decoder ────────────────────────────────────────────────────────

    def decode(
        self,
        wrist_img: torch.Tensor,   # (B, 3, H, W)
        stand_img: torch.Tensor,   # (B, 3, H, W)
        proprio:   torch.Tensor,   # (B, action_dim)
        z:         torch.Tensor,   # (B, z_dim)
    ) -> torch.Tensor:
        B = wrist_img.shape[0]

        # Visual features as memory tokens
        w_feat = self.wrist_proj(self.wrist_encoder(wrist_img))   # (B, d)
        s_feat = self.stand_proj(self.stand_encoder(stand_img))   # (B, d)
        memory = torch.stack([w_feat, s_feat], dim=1)             # (B, 2, d)

        # Also add proprio and z to memory
        p_tok  = self.proprio_dec_fc(proprio).unsqueeze(1)        # (B, 1, d)
        z_tok  = self.z_proj(z).unsqueeze(1)                      # (B, 1, d)
        memory = torch.cat([memory, p_tok, z_tok], dim=1)         # (B, 4, d)

        # Queries = learned positional embeddings for each action step
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # (B,T,d)

        out = self.transformer_dec(queries, memory)                # (B, T, d)
        return self.action_head(out)                               # (B, T, action_dim)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        wrist_img: torch.Tensor,
        stand_img: torch.Tensor,
        proprio:   torch.Tensor,
        actions:   torch.Tensor,   # (B, chunk_size, action_dim)
        beta:      float = 10.0,   # KL weight
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with 'loss', 'loss_recon', 'loss_kl'."""
        # Normalise actions
        nactions = (actions - self.action_mean) / (self.action_std + 1e-8)

        mu, log_var = self.encode_style(proprio, nactions)
        z           = self.reparameterise(mu, log_var)

        pred_actions = self.decode(wrist_img, stand_img, proprio, z)

        loss_recon = F.l1_loss(pred_actions, nactions)
        loss_kl    = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

        loss = loss_recon + beta * loss_kl
        return {"loss": loss, "loss_recon": loss_recon, "loss_kl": loss_kl}

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_action(
        self,
        wrist_img: torch.Tensor,   # (1, 3, H, W)
        stand_img: torch.Tensor,
        proprio:   torch.Tensor,   # (1, action_dim)
        use_ensemble: bool = True,
    ) -> np.ndarray:
        """
        Returns next action (action_dim,) after temporal ensemble.
        """
        self.eval()
        z = torch.zeros(1, self.z_dim, device=self.device)

        pred_norm = self.decode(wrist_img, stand_img, proprio, z)  # (1,T,A)
        pred      = (pred_norm * (self.action_std + 1e-8) + self.action_mean)
        pred_np   = pred.squeeze(0).cpu().numpy()                   # (T, A)

        if not use_ensemble:
            return pred_np[0]

        # Temporal ensemble: push new prediction, pop old, compute weighted avg
        self._ensemble_buffer.append(pred_np)
        # Keep only as many predictions as chunk_size
        if len(self._ensemble_buffer) > self.chunk_size:
            self._ensemble_buffer.pop(0)

        # Each prediction at offset k contributes its k-th action
        # to the current step, weighted by exp decay
        actions_at_t0 = []
        weights_t0    = []
        for offset, chunk in enumerate(self._ensemble_buffer):
            idx = offset   # How many steps into this chunk corresponds to "now"
            if idx < len(chunk):
                actions_at_t0.append(chunk[idx])
                weights_t0.append(self._ensemble_weights[idx])

        weights_t0  = np.array(weights_t0)
        weights_t0 /= weights_t0.sum()
        return np.average(actions_at_t0, axis=0, weights=weights_t0)

    def reset_ensemble(self) -> None:
        self._ensemble_buffer.clear()

    # ── Persistence ───────────────────────────────────────────────────────────

    def set_normalisation_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.action_mean = torch.from_numpy(mean).float().to(self.device)
        self.action_std  = torch.from_numpy(std).float().to(self.device)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info("ACTPolicy saved to %s", path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("ACTPolicy loaded from %s", path)
