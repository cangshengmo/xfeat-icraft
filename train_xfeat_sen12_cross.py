"""
Cross-modal training on SEN1-2 dataset (perfectly aligned SAR/OPT pairs).

The identity homography gives dense pixel-level correspondences,
providing the strongest possible supervision for cross-modal descriptor learning.

Usage:
  python train_xfeat_sen12_cross.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

from modules.model import XFeatModel
from modules.training.losses import (
    _HAVE_ALIKE,
    dual_softmax_loss,
    keypoint_loss,
)
from modules.training.utils import check_accuracy, get_nb_trainable_params

if _HAVE_ALIKE:
    from modules.training.losses import alike_distill_loss

from modules.dataset.saropt_dataset import SEN12CrossModalDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"D:\HanZhQ\data\SEN1-2_256\train"))
    p.add_argument("--val-root", type=Path, default=Path(r"D:\HanZhQ\data\SEN1-2_256\val"))
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--ckpt-save-path", type=Path, default=Path("./checkpoints"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--lr-decay-steps", type=int, default=8)
    p.add_argument("--resolution", type=str, default="256,256")
    p.add_argument("--preprocess", choices=["raw", "grad"], default="grad")
    p.add_argument("--device-num", type=str, default="0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


class Trainer:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        self.args = args
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SEN12Cross] Device: {self.dev}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = args.ckpt_save_path / f"xfeat-sen12-cross-{ts}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SEN12Cross] Output: {self.out_dir}")

        # Network
        self.net = XFeatModel().to(self.dev)
        w = args.weights or str(Path(__file__).resolve().parent / "weights" / "xfeat.pt")
        if Path(w).exists():
            print(f"[SEN12Cross] Loading: {w}")
            self.net.load_state_dict(torch.load(w, map_location=self.dev))

        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=args.lr)

        # Datasets
        res = tuple(int(v) for v in args.resolution.split(","))
        train_ds = SEN12CrossModalDataset(args.data_root, resolution=res, preprocess=args.preprocess, limit=args.limit)
        val_ds = SEN12CrossModalDataset(args.val_root, resolution=res, preprocess=args.preprocess, limit=args.limit or 500)

        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        self.img_size = (res[1], res[0])  # (H, W)
        self.writer = SummaryWriter(str(self.out_dir / "tensorboard"))
        self.epochs = args.epochs
        self.dry_run = args.dry_run
        self.best_loss = float("inf")

        print(f"[SEN12Cross] Train: {len(train_ds)}  Val: {len(val_ds)}  Epochs: {self.epochs}")
        # StepLR step_size is in optimizer steps: 1 epoch ≈ len//batch_size steps
        # Decay lr every 10 epochs
        steps_per_epoch = len(train_ds) // args.batch_size
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=steps_per_epoch * 10, gamma=args.gamma)
        get_nb_trainable_params(self.net)

    def _sample_corrs(self, batch_size, device):
        """Generate dense grid of identity correspondences."""
        H, W = self.img_size
        h_g, w_g = H // 8, W // 8
        margin = 4
        ys = torch.arange(margin, h_g - margin, device=device)
        xs = torch.arange(margin, w_g - margin, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1).float() * 8
        pts = pts.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 2)
        return pts, pts.clone()  # identity

    def _save_best(self, loss):
        if loss >= self.best_loss:
            return
        self.best_loss = loss
        torch.save({"model_state": self.net.state_dict(), "loss": loss}, str(self.out_dir / "best.pth"))
        print(f"\n[SEN12Cross] ★ New best: {loss:.4f}")

    def train_epoch(self, epoch):
        self.net.train()
        losses, accs_c, accs_kp = [], [], []
        global_step = epoch * len(self.train_loader)

        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for sars, opts, _ in pbar:
            B = sars.shape[0]
            sars, opts = sars.to(self.dev), opts.to(self.dev)

            pts1, pts2 = self._sample_corrs(B, self.dev)

            feats1, kpts1, hmap1 = self.net(sars)
            feats2, kpts2, hmap2 = self.net(opts)

            loss_items = []
            acc_c_list, acc_kp_list = [], []

            for b in range(B):
                p1 = pts1[b]
                p2 = pts2[b]
                # Feature-space coordinates (÷8)
                py = (p1[:, 1] / 8).long().clamp(0, feats1.shape[2] - 1)
                px = (p1[:, 0] / 8).long().clamp(0, feats1.shape[3] - 1)

                m1 = feats1[b, :, py, px].permute(1, 0)
                m2 = feats2[b, :, py, px].permute(1, 0)
                h1 = hmap1[b, 0, py, px]
                h2 = hmap2[b, 0, py, px]

                loss_ds, conf = dual_softmax_loss(m1, m2)
                loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)
                loss_items += [loss_ds.unsqueeze(0), loss_kp.unsqueeze(0)]

                if _HAVE_ALIKE:
                    l1, a1 = alike_distill_loss(kpts1[b], sars[b])
                    l2, a2 = alike_distill_loss(kpts2[b], opts[b])
                    loss_items += [l1.unsqueeze(0), l2.unsqueeze(0)]
                    acc_kp_list.append((a1 + a2) / 2)

                acc_c_list.append(check_accuracy(m1, m2))

            if not loss_items:
                continue

            loss = torch.cat(loss_items).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()
            self.scheduler.step()

            avg_c = float(np.mean([v.cpu() if torch.is_tensor(v) else v for v in acc_c_list])) if acc_c_list else 0
            avg_kp = float(np.mean([v.cpu() if torch.is_tensor(v) else v for v in acc_kp_list])) if acc_kp_list else 0

            losses.append(loss.item())
            accs_c.append(avg_c)
            accs_kp.append(avg_kp)
            global_step += 1

            pbar.set_description(f"Loss: {loss.item():.4f}  acc_c: {avg_c:.3f}  acc_kp: {avg_kp:.3f}")
            self.writer.add_scalar("train/loss", loss.item(), global_step)
            self.writer.add_scalar("train/acc_c", avg_c, global_step)
            self.writer.add_scalar("train/acc_kp", avg_kp, global_step)

            if self.dry_run and global_step >= 20:
                break

        return float(np.mean(losses)), float(np.mean(accs_c)), float(np.mean(accs_kp))

    def train(self):
        print(f"[SEN12Cross] Starting {self.epochs} epochs ...")
        for epoch in range(self.epochs):
            loss, acc_c, acc_kp = self.train_epoch(epoch)

            # Monitoring check
            status = f"Epoch {epoch}: loss={loss:.4f}  acc_c={acc_c:.4f}  acc_kp={acc_kp:.4f}  lr={self.scheduler.get_last_lr()[0]:.6f}"
            print(status)
            with open(self.out_dir / "monitor.log", "a") as f:
                f.write(status + "\n")

            self._save_best(loss)

            if self.dry_run and epoch >= 1:
                break

        self.writer.close()
        print(f"[SEN12Cross] Done. Best loss: {self.best_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    Trainer(args).train()
