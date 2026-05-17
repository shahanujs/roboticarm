"""
Training script for Diffusion Policy and ACT on SO-ARM101 data.

Usage
-----
# Train Diffusion Policy
python train.py --policy diffusion --data_dir datasets/ --epochs 300

# Train ACT
python train.py --policy act --data_dir datasets/ --epochs 200

Checkpoints are saved to checkpoints/{policy}_ep{N}.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import RobotPickDataset
from policy.act_policy import ACTPolicy
from policy.diffusion_policy import DiffusionPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def get_device() -> str:
    if torch.cuda.is_available():
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return "cuda"
    logger.warning("CUDA not available — using CPU (slow!)")
    return "cpu"


def train_diffusion(args, device: str) -> None:
    dataset = RobotPickDataset(
        data_dir=args.data_dir,
        obs_horizon=2,
        pred_horizon=16,
        augment=True,
    )

    val_size  = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)

    policy = DiffusionPolicy(
        action_dim=7, obs_horizon=2, pred_horizon=16,
        action_horizon=8, diffusion_steps=100, device=device,
    )
    policy.set_normalisation_stats(dataset.action_mean, dataset.action_std)

    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        policy.train()
        t0 = time.monotonic()
        train_losses = []
        for batch in train_loader:
            wrist  = batch["wrist_imgs"].to(device)
            stand  = batch["stand_imgs"].to(device)
            proprio = batch["proprio"].to(device)
            actions = batch["actions"].to(device)
            # Denormalise for forward (policy normalises internally)
            actions_orig = actions * torch.from_numpy(dataset.action_std).to(device) \
                           + torch.from_numpy(dataset.action_mean).to(device)

            optimiser.zero_grad()
            loss = policy(wrist, stand, proprio, actions_orig)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        policy.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                wrist   = batch["wrist_imgs"].to(device)
                stand   = batch["stand_imgs"].to(device)
                proprio  = batch["proprio"].to(device)
                actions  = batch["actions"].to(device)
                actions_orig = actions * torch.from_numpy(dataset.action_std).to(device) \
                               + torch.from_numpy(dataset.action_mean).to(device)
                val_losses.append(policy(wrist, stand, proprio, actions_orig).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        elapsed    = time.monotonic() - t0

        logger.info("Ep %3d/%d  train=%.5f  val=%.5f  lr=%.2e  t=%.1fs",
                    epoch, args.epochs, train_loss, val_loss,
                    scheduler.get_last_lr()[0], elapsed)

        # ── Save checkpoints ──────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            ckpt = os.path.join(args.ckpt_dir, f"diffusion_ep{epoch:04d}.pth")
            policy.save(ckpt)

        if val_loss < best_val:
            best_val = val_loss
            policy.save(os.path.join(args.ckpt_dir, "diffusion_best.pth"))

    logger.info("Diffusion training done. Best val loss: %.5f", best_val)


def train_act(args, device: str) -> None:
    dataset = RobotPickDataset(
        data_dir=args.data_dir,
        obs_horizon=1,           # ACT uses single-frame obs
        pred_horizon=50,         # ACT standard chunk size
        augment=True,
    )

    val_size  = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)

    policy = ACTPolicy(
        action_dim=7, chunk_size=50, d_model=512, nhead=8,
        num_enc_layers=4, num_dec_layers=7, z_dim=32, device=device,
    )
    policy.set_normalisation_stats(dataset.action_mean, dataset.action_std)

    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        t0 = time.monotonic()
        train_losses = []

        for batch in train_loader:
            # ACT obs_horizon=1 → squeeze time dim
            wrist   = batch["wrist_imgs"].squeeze(1).to(device)   # (B,3,H,W)
            stand   = batch["stand_imgs"].squeeze(1).to(device)
            proprio  = batch["proprio"].squeeze(1).to(device)       # (B,7)
            actions  = batch["actions"].to(device)                  # (B,50,7)
            actions_orig = actions * torch.from_numpy(dataset.action_std).to(device) \
                           + torch.from_numpy(dataset.action_mean).to(device)

            optimiser.zero_grad()
            result = policy(wrist, stand, proprio, actions_orig, beta=10.0)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            train_losses.append(result["loss"].item())

        scheduler.step()

        policy.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                wrist   = batch["wrist_imgs"].squeeze(1).to(device)
                stand   = batch["stand_imgs"].squeeze(1).to(device)
                proprio  = batch["proprio"].squeeze(1).to(device)
                actions  = batch["actions"].to(device)
                actions_orig = actions * torch.from_numpy(dataset.action_std).to(device) \
                               + torch.from_numpy(dataset.action_mean).to(device)
                val_losses.append(policy(wrist, stand, proprio, actions_orig)["loss"].item())

        logger.info("Ep %3d/%d  train=%.4f  val=%.4f  lr=%.2e  t=%.1fs",
                    epoch, args.epochs, np.mean(train_losses), np.mean(val_losses),
                    scheduler.get_last_lr()[0], time.monotonic() - t0)

        if epoch % args.save_every == 0:
            policy.save(os.path.join(args.ckpt_dir, f"act_ep{epoch:04d}.pth"))
        if np.mean(val_losses) < best_val:
            best_val = np.mean(val_losses)
            policy.save(os.path.join(args.ckpt_dir, "act_best.pth"))

    logger.info("ACT training done. Best val loss: %.5f", best_val)


def main():
    parser = argparse.ArgumentParser(description="Train robot manipulation policy")
    parser.add_argument("--policy",     choices=["diffusion", "act"], default="diffusion")
    parser.add_argument("--data_dir",   type=str, default="datasets")
    parser.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    parser.add_argument("--epochs",     type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    device = get_device()

    if args.policy == "diffusion":
        train_diffusion(args, device)
    else:
        train_act(args, device)


if __name__ == "__main__":
    main()
