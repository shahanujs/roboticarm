"""
Dataset loader for SO-ARM101 diffusion policy training.
Loads episodes from datasets/*.npz, constructs (obs, action) pairs.

Observation:
  wrist_imgs  : (obs_horizon, 3, H, W)  float32 normalised [0,1]
  stand_imgs  : (obs_horizon, 3, H, W)
  proprio     : (obs_horizon, action_dim)

Action (label):
  actions     : (pred_horizon, action_dim)   normalised
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

ACTION_DIM  = 7   # 6 joints + 1 gripper
IMG_HEIGHT  = 96
IMG_WIDTH   = 96


def load_episodes(data_dir: str) -> List[Dict[str, np.ndarray]]:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")
    episodes = []
    for p in paths:
        ep = dict(np.load(p))
        episodes.append(ep)
    logger.info("Loaded %d episodes from %s", len(episodes), data_dir)
    return episodes


def compute_normalisation_stats(episodes: List[Dict[str, np.ndarray]]
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean and std over all actions in dataset."""
    all_actions = []
    for ep in episodes:
        joints = ep["act_joints"]           # (T, 6)
        gripper = ep["act_gripper"][:, None] # (T, 1)
        actions = np.concatenate([joints, gripper], axis=-1)
        all_actions.append(actions)
    all_actions = np.concatenate(all_actions, axis=0)   # (N, 7)
    mean = all_actions.mean(axis=0).astype(np.float32)
    std  = all_actions.std(axis=0).astype(np.float32) + 1e-6
    return mean, std


def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) → RGB float32 (3,H,W) normalised [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32).transpose(2, 0, 1) / 255.0


class RobotPickDataset(Dataset):
    """
    PyTorch dataset for robot manipulation episodes.
    Each item = one (obs, action_chunk) pair sampled from an episode.

    Parameters
    ----------
    data_dir      : path to directory containing episode_*.npz
    obs_horizon   : how many past frames to stack as observation
    pred_horizon  : how many future actions to predict
    augment       : apply random colour jitter augmentation
    """

    def __init__(
        self,
        data_dir:    str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        augment:     bool = True,
        action_mean: Optional[np.ndarray] = None,
        action_std:  Optional[np.ndarray] = None,
    ):
        self.obs_horizon  = obs_horizon
        self.pred_horizon = pred_horizon
        self.augment      = augment

        episodes = load_episodes(data_dir)

        if action_mean is None or action_std is None:
            action_mean, action_std = compute_normalisation_stats(episodes)
        self.action_mean = action_mean
        self.action_std  = action_std

        # Flatten into (episode_idx, start_idx) index
        self.index: List[Tuple[int, int]] = []
        self.episodes = []

        for ep in episodes:
            T = ep["act_joints"].shape[0]
            # Need at least obs_horizon past + pred_horizon future
            for t in range(obs_horizon - 1, T - pred_horizon + 1):
                self.index.append((len(self.episodes), t))
            self.episodes.append(ep)

        logger.info("Dataset: %d samples from %d episodes", len(self.index), len(self.episodes))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]

        # ── Observation ───────────────────────────────────────────────────────
        wrist_imgs = []
        stand_imgs = []
        props      = []

        for h in range(self.obs_horizon - 1, -1, -1):
            ti = max(0, t - h)
            wrist_imgs.append(preprocess_image(ep["wrist_imgs"][ti]))
            stand_imgs.append(preprocess_image(ep["stand_imgs"][ti]))
            joints  = ep["joints_deg"][ti].astype(np.float32)
            gripper = np.array([ep["gripper_w"][ti]], dtype=np.float32)
            props.append(np.concatenate([joints, gripper]))

        wrist_imgs = np.stack(wrist_imgs)   # (obs_h, 3, H, W)
        stand_imgs = np.stack(stand_imgs)
        proprio    = np.stack(props)         # (obs_h, action_dim)

        # ── Action chunk ──────────────────────────────────────────────────────
        act_list = []
        for i in range(self.pred_horizon):
            ti = min(t + i, ep["act_joints"].shape[0] - 1)
            j  = ep["act_joints"][ti].astype(np.float32)
            g  = np.array([ep["act_gripper"][ti]], dtype=np.float32)
            act_list.append(np.concatenate([j, g]))
        actions = np.stack(act_list)         # (pred_horizon, action_dim)

        # Normalise actions
        actions = (actions - self.action_mean) / self.action_std

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            wrist_imgs = self._augment_imgs(wrist_imgs)
            stand_imgs = self._augment_imgs(stand_imgs)

        return {
            "wrist_imgs": torch.from_numpy(wrist_imgs),
            "stand_imgs": torch.from_numpy(stand_imgs),
            "proprio":    torch.from_numpy(proprio),
            "actions":    torch.from_numpy(actions),
        }

    @staticmethod
    def _augment_imgs(imgs: np.ndarray) -> np.ndarray:
        """Random colour jitter on (T, 3, H, W) float32 images in [0,1]."""
        # Brightness / contrast
        alpha = np.random.uniform(0.85, 1.15)
        beta  = np.random.uniform(-0.10, 0.10)
        return (imgs * alpha + beta).clip(0, 1).astype(np.float32)
