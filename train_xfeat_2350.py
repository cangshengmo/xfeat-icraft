"""
Fine-tune XFeat on the 2350 SAR/OPT dataset using self-supervised warp training.

Strategy:
  - Load pretrained XFeat weights
  - Use SAROptAugmentationPipe to generate random warped pairs from 2350 images
  - Train with the same losses as the original XFeat (dual_softmax_loss,
    coordinate_classification_loss, keypoint_loss, alike_distill_loss)
  - Save checkpoints for downstream eval

Usage:
  # Dry-run (single batch sanity check)
  python train_xfeat_2350.py --dry-run

  # Full fine-tune
  python train_xfeat_2350.py ^
      --data-root D:\\HanZhQ\\PCIE715\\Project_Trans\\2350 ^
      --ckpt-save-path ./checkpoints/2350_warp ^
      --batch-size 8 ^
      --n-steps 20000 ^
      --lr 3e-4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import tqdm

from modules.model import XFeatModel
from modules.training.losses import (
    alike_distill_loss,
    check_accuracy,
    coordinate_classification_loss,
    dual_softmax_loss,
    keypoint_loss,
)
from modules.training.utils import get_corresponding_pts, get_nb_trainable_params, make_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune XFeat on 2350 SAR/OPT data.")
    p.add_argument("--data-root", type=Path, default=Path(r"D:\HanZhQ\PCIE715\Project_Trans\2350"),
                   help="2350 dataset root with sar/ opt/ label.txt")
    p.add_argument("--weights", type=str, default=None,
                   help="Initial weights path. Default: weights/xfeat.pt")
    p.add_argument("--ckpt-save-path", type=Path, default=Path("./checkpoints/xfeat_2350"),
                   help="Checkpoint and log directory")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p.add_argument("--n-steps", type=int, default=20_000, help="Number of training steps")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma-steplr", type=float, default=0.5, help="StepLR gamma")
    p.add_argument("--training-res", type=str, default="800,608",
                   help="Training resolution (width,height)")
    p.add_argument("--device-num", type=str, default="0", help="CUDA device number")
    p.add_argument("--dry-run", action="store_true", help="Single mini-batch sanity check")
    p.add_argument("--save-ckpt-every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--max-imgs", type=int, default=500,
                   help="Max training images to load from 2350")
    return p.parse_args()


class Trainer2350:
    """Fine-tune XFeat on 2350 data via self-supervised warp augmentation."""

    def __init__(self, args: argparse.Namespace):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        self.args = args
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer2350] Device: {self.dev}")

        # ── Network ────────────────────────────────────────────────
        self.net = XFeatModel().to(self.dev)

        # Load pretrained weights
        weights_path = args.weights or (Path(__file__).resolve().parent / "weights" / "xfeat.pt")
        if weights_path.exists():
            print(f"[Trainer2350] Loading pretrained weights from: {weights_path}")
            self.net.load_state_dict(torch.load(str(weights_path), map_location=self.dev))
        else:
            print(f"[Trainer2350] WARNING: weights not found at {weights_path}, training from scratch")

        # ── Optimiser ──────────────────────────────────────────────
        self.opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), lr=args.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=args.gamma_steplr)

        # ── Augmentation pipe (self-supervised warps from 2350 images) ──
        training_res = tuple(int(v) for v in args.training_res.split(","))
        from modules.dataset.saropt_dataset import SAROptAugmentationPipe

        self.augmentor = SAROptAugmentationPipe(
            data_root=args.data_root,
            device=self.dev,
            batch_size=args.batch_size,
            warp_resolution=training_res,
            out_resolution=training_res,
            max_num_imgs=args.max_imgs,
            sides_crop=0.1,
            photometric=True,
            geometric=True,
            reload_step=4_000,
        )
        self.training_res = training_res

        # ── Misc ───────────────────────────────────────────────────
        args.ckpt_save_path.mkdir(parents=True, exist_ok=True)
        logdir = args.ckpt_save_path / "logdir"
        logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            str(logdir / f"xfeat_2350_{time.strftime('%Y_%m_%d-%H_%M_%S')}")
        )
        self.steps = args.n_steps
        self.save_ckpt_every = args.save_ckpt_every
        self.dry_run = args.dry_run

        get_nb_trainable_params(self.net)

    def train(self):
        self.net.train()
        difficulty = 0.10

        print(f"[Trainer2350] Starting training for {self.steps} steps ...")
        pbar = tqdm.tqdm(total=self.steps)

        for i in range(self.steps):
            # ── Grab synthetic warp batch ──────────────────────────
            p1, p2, H1, H2 = make_batch(self.augmentor, difficulty)

            # ── Convert to grayscale ───────────────────────────────
            p1 = p1.mean(1, keepdim=True)
            p2 = p2.mean(1, keepdim=True)

            # ── Get ground-truth correspondences from warp params ──
            h_coarse, w_coarse = p1.shape[-2] // 8, p1.shape[-1] // 8
            _, positives = get_corresponding_pts(
                p1, p2, H1, H2, self.augmentor, h_coarse, w_coarse
            )

            # ── Skip corrupted batches ─────────────────────────────
            is_corrupted = any(len(p) < 30 for p in positives)
            if is_corrupted:
                continue

            # ── Forward ────────────────────────────────────────────
            feats1, kpts1, hmap1 = self.net(p1)
            feats2, kpts2, hmap2 = self.net(p2)

            loss_items = []
            total_acc_coarse = 0.0
            total_acc_fine = 0.0
            total_acc_pos = 0.0
            nb = len(positives)

            for b in range(nb):
                pts1, pts2 = positives[b][:, :2], positives[b][:, 2:]

                # Sample descriptors at keypoint positions
                m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
                m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)

                # Reliability maps
                h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]

                # Fine coordinate refinement
                coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                # ── Losses ─────────────────────────────────────────
                loss_ds, conf = dual_softmax_loss(m1, m2)
                loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                acc_pos = (acc_pos1 + acc_pos2) / 2.0

                loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                loss_items += [
                    loss_ds.unsqueeze(0),
                    loss_coords.unsqueeze(0),
                    loss_kp.unsqueeze(0),
                    loss_kp_pos.unsqueeze(0),
                ]

                acc_coarse = check_accuracy(m1, m2)
                total_acc_coarse += acc_coarse
                total_acc_fine += acc_coords
                total_acc_pos += acc_pos

            # ── Backward ───────────────────────────────────────────
            loss = torch.cat(loss_items, -1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()
            self.scheduler.step()

            # ── Logging ────────────────────────────────────────────
            avg_acc_coarse = total_acc_coarse / nb
            avg_acc_fine = total_acc_fine / nb
            avg_acc_pos = total_acc_pos / nb

            pbar.set_description(
                f"Loss: {loss.item():.4f}  acc_c: {avg_acc_coarse:.3f}  "
                f"acc_f: {avg_acc_fine:.3f}  acc_kp: {avg_acc_pos:.3f}"
            )
            pbar.update(1)

            self.writer.add_scalar("Loss/total", loss.item(), i)
            self.writer.add_scalar("Accuracy/coarse", avg_acc_coarse, i)
            self.writer.add_scalar("Accuracy/fine", avg_acc_fine, i)
            self.writer.add_scalar("Accuracy/keypoint_position", avg_acc_pos, i)

            # ── Save checkpoint ────────────────────────────────────
            if (i + 1) % self.save_ckpt_every == 0 and not self.dry_run:
                ckpt_path = self.args.ckpt_save_path / f"xfeat_2350_{i+1}.pth"
                torch.save(self.net.state_dict(), str(ckpt_path))
                print(f"\n[Trainer2350] Checkpoint saved: {ckpt_path}")

            if self.dry_run and i >= 2:
                print("[Trainer2350] Dry-run complete (3 steps).")
                break

        pbar.close()
        print("[Trainer2350] Training complete.")


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer2350(args)
    trainer.train()
