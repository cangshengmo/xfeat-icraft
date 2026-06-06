"""
Cross-modal weakly-supervised fine-tune of XFeat on 2350 SAR/OPT pairs.

Uses label.txt translation to generate dense correspondences between SAR and OPT
images, then trains descriptor consistency and keypoint quality across modalities.

Usage:
  # Dry-run
  python train_xfeat_2350_cross.py --dry-run

  # Full training
  python train_xfeat_2350_cross.py ^
      --data-root D:\\HanZhQ\\PCIE715\\Project_Trans\\2350 ^
      --batch-size 8 --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
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


# ── Dataset ─────────────────────────────────────────────────────────

class CrossModalDataset(Dataset):
    """Load SAR/OPT pairs from 2350 and generate dense translation correspondences."""

    def __init__(
        self,
        data_root: str | Path,
        resolution: tuple[int, int] = (512, 512),
        preprocess: str = "grad",
        limit: int = 0,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.data_root = Path(data_root)
        self.resolution = resolution  # (W, H)
        self.preprocess = preprocess
        self.split = split

        labels = self._read_labels()
        n = len(labels)
        n_train = int(n * train_ratio)
        if split == "train":
            self.labels = labels[:n_train]
        else:
            self.labels = labels[n_train:]

        if limit > 0:
            self.labels = self.labels[:limit]

        print(f"[CrossModal] {split}: {len(self.labels)} pairs")

    def _read_labels(self):
        rows = []
        for line in (self.data_root / "label.txt").read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 3:
                rows.append({"id": p[0], "x": float(p[1]), "y": float(p[2])})
        return rows

    def _find_image(self, subdir, image_id):
        folder = self.data_root / subdir
        for ext in (".bmp", ".png", ".jpg", ".jpeg"):
            p = folder / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{folder / image_id}.*")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.preprocess == "raw":
            return img
        if self.preprocess == "grad":
            blur = cv2.GaussianBlur(img, (3, 3), 0)
            gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.preprocess == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img)
        return img

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels[idx]

        # Load images
        sar = cv2.imread(str(self._find_image("sar", row["id"])), cv2.IMREAD_GRAYSCALE)
        opt = cv2.imread(str(self._find_image("opt", row["id"])), cv2.IMREAD_GRAYSCALE)

        # Preprocess
        sar = self._preprocess(sar)   # (512, 512)
        opt = self._preprocess(opt)   # (800, 800)

        # Translation from label.txt
        tx, ty = row["x"], row["y"]

        # Resize both to training resolution
        W, H = self.resolution
        sar = cv2.resize(sar, (W, H))
        opt = cv2.resize(opt, (W, H))

        # Scale translation to training resolution
        # SAR: 512→W, OPT: 800→W (aspect-ratio preserved)
        sx = W / 512.0
        sy = H / 512.0
        ox = W / 800.0
        oy = H / 800.0
        tx_scaled = tx * ox
        ty_scaled = ty * oy
        sar_to_opt_scale = ox / sx  # 512/800 = 0.64: SAR→OPT scale factor

        # Build homography: SAR point (xs, ys) → OPT point (xs*scale + tx, ys*scale + ty)
        H_mat = np.array([
            [sar_to_opt_scale, 0, tx_scaled],
            [0, sar_to_opt_scale, ty_scaled],
            [0, 0, 1]
        ], dtype=np.float32)

        sar_t = torch.from_numpy(sar).float().unsqueeze(0) / 255.0   # (1, H, W)
        opt_t = torch.from_numpy(opt).float().unsqueeze(0) / 255.0
        H_t = torch.from_numpy(H_mat)

        return sar_t, opt_t, H_t


def collate_crossmodal(batch):
    sars, opts, Hs = zip(*batch)
    return torch.stack(sars), torch.stack(opts), torch.stack(Hs)


# ── Correspondence helpers ──────────────────────────────────────────

@torch.inference_mode()
def sample_correspondences(H_batch, img_size, device, stride=8, margin=8):
    """
    Given homography (B, 3, 3) from SAR→OPT, sample dense grid in SAR space
    and map to OPT space. Returns valid correspondences within bounds.
    """
    B = H_batch.shape[0]
    H_img, W_img = img_size
    h_grid = H_img // 8
    w_grid = W_img // 8

    # Grid in SAR feature space (downsampled 8x)
    ys = torch.arange(margin, h_grid - margin, device=device)
    xs = torch.arange(margin, w_grid - margin, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid_pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1).float()  # (N, 2)

    # Map to image coordinates
    pts_sar = grid_pts * 8  # (N, 2) in SAR image space
    N = len(pts_sar)

    results = []
    for b in range(B):
        H = H_batch[b]  # (3, 3)
        # Transform SAR points to OPT
        ones = torch.ones(N, 1, device=device)
        sar_homo = torch.cat([pts_sar, ones], dim=-1)  # (N, 3)
        opt_homo = (H @ sar_homo.T).T  # (N, 3)
        pts_opt = opt_homo[:, :2] / opt_homo[:, 2:3]  # (N, 2)

        # Filter in-bounds
        valid = (
            (pts_opt[:, 0] >= 0) & (pts_opt[:, 0] < W_img) &
            (pts_opt[:, 1] >= 0) & (pts_opt[:, 1] < H_img)
        )
        if valid.sum() < 10:
            results.append((torch.zeros(0, 2, device=device), torch.zeros(0, 2, device=device)))
            continue

        sar_valid = pts_sar[valid]
        opt_valid = pts_opt[valid]

        # Subsample if too many
        max_pts = 2048
        if len(sar_valid) > max_pts:
            perm = torch.randperm(len(sar_valid), device=device)[:max_pts]
            sar_valid = sar_valid[perm]
            opt_valid = opt_valid[perm]

        results.append((sar_valid, opt_valid))

    return results  # list of (sar_pts, opt_pts) tuples


# ── Trainer ─────────────────────────────────────────────────────────

class CrossModalTrainer:
    def __init__(self, args: argparse.Namespace):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        self.args = args
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CrossModal] Device: {self.dev}")

        # Output directory
        if args.resume:
            self.out_dir = Path(args.resume).resolve().parent
            self.out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CrossModal] Resuming to same output dir: {self.out_dir}")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"xfeat-2350-cross-{ts}"
            self.out_dir = args.ckpt_save_path / run_name
            self.out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CrossModal] Output: {self.out_dir}")

        # Network
        self.net = XFeatModel().to(self.dev)
        weights_path = args.weights or (Path(__file__).resolve().parent / "weights" / "xfeat.pt")
        if weights_path.exists():
            print(f"[CrossModal] Loading weights: {weights_path}")
            self.net.load_state_dict(torch.load(str(weights_path), map_location=self.dev))

        # Optimizer
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=args.lr_decay_steps, gamma=args.gamma)

        # Resume from checkpoint
        if args.resume:
            print(f"[CrossModal] Resuming from {args.resume}")
            ckpt = torch.load(str(args.resume), map_location=self.dev)
            if "model_state" in ckpt:
                self.net.load_state_dict(ckpt["model_state"])
            else:
                self.net.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                self.opt.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
                print(f"[CrossModal] Restored scheduler (last_epoch={self.scheduler.last_epoch})")

        # Dataset
        train_ds = CrossModalDataset(
            data_root=args.data_root,
            resolution=tuple(int(v) for v in args.resolution.split(",")),
            preprocess=args.preprocess,
            limit=args.limit,
            split="train",
        )
        val_ds = CrossModalDataset(
            data_root=args.data_root,
            resolution=tuple(int(v) for v in args.resolution.split(",")),
            preprocess=args.preprocess,
            limit=max(args.limit // 5, 10) if args.limit else 50,
            split="val",
        )
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_crossmodal)
        self.val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_crossmodal)

        self.img_size = tuple(int(v) for v in args.resolution.split(","))  # (W, H) → (H, W) for internal use
        self.img_size = (self.img_size[1], self.img_size[0])  # (H, W)

        # Logging
        self.writer = SummaryWriter(str(self.out_dir / "tensorboard"))
        self.csv_path = self.out_dir / "training_log.csv"
        self.history = []
        self.best_loss = float("inf")
        self.epochs = args.epochs
        self.start_epoch = 0
        self.dry_run = args.dry_run

        if args.resume:
            # Determine start epoch from existing CSV
            if self.csv_path.exists():
                with open(self.csv_path, newline="") as f:
                    reader = list(csv.reader(f))
                if len(reader) > 1:
                    last_row = reader[-1]
                    self.start_epoch = int(last_row[0]) + 1
                    print(f"[CrossModal] Found {len(reader)-1} logged steps, resuming from epoch {self.start_epoch}")
            # Read best_loss from best.pth if available
            best_path = self.out_dir / "best.pth"
            if best_path.exists():
                best_ckpt = torch.load(str(best_path), map_location=self.dev)
                self.best_loss = best_ckpt.get("loss", float("inf"))
                print(f"[CrossModal] Restored best_loss={self.best_loss:.4f}")
        else:
            self._init_csv()

        print(f"[CrossModal] Train: {len(train_ds)}  Val: {len(val_ds)}  Epochs: {self.epochs}")
        get_nb_trainable_params(self.net)

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "step", "loss", "acc_c", "acc_kp"])

    def _log(self, epoch, step, loss, acc_c, acc_kp):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, step, f"{loss:.6f}", f"{acc_c:.4f}", f"{acc_kp:.4f}"])

    def _save_best(self, epoch, loss):
        if loss >= self.best_loss:
            return
        self.best_loss = loss
        torch.save({"epoch": epoch, "model_state": self.net.state_dict(), "loss": loss},
                    str(self.out_dir / "best.pth"))
        print(f"\n[CrossModal] ★ New best: loss={loss:.4f}")

    def _save_latest(self):
        torch.save({
            "epoch": self.current_epoch,
            "model_state": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, str(self.out_dir / "latest.pth"))

    def _plot_curves(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if len(self.history) < 3:
            return
        steps = [h["step"] for h in self.history]
        _, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(steps, [h["loss"] for h in self.history], label="Loss", color="tab:red")
        ax[0].set_xlabel("Step"); ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(steps, [h["acc_c"] for h in self.history], label="acc_c", color="tab:blue")
        ax[1].plot(steps, [h["acc_kp"] for h in self.history], label="acc_kp", color="tab:green")
        ax[1].set_xlabel("Step"); ax[1].set_ylabel("Accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(self.out_dir / "curves.png"), dpi=150)
        plt.close()

    def train_epoch(self, epoch):
        self.net.train()
        total_loss, total_acc_c, total_acc_kp, n_batch = 0, 0, 0, 0
        global_step = epoch * len(self.train_loader)

        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for sars, opts, Hs in pbar:
            sars, opts, Hs = sars.to(self.dev), opts.to(self.dev), Hs.to(self.dev)
            B = sars.shape[0]

            # Get correspondences
            corrs = sample_correspondences(Hs, self.img_size, self.dev, stride=8, margin=8)

            # Forward
            feats1, kpts1, hmap1 = self.net(sars)
            feats2, kpts2, hmap2 = self.net(opts)

            loss_items = []
            acc_c_list, acc_kp_list = [], []

            for b in range(B):
                pts_sar, pts_opt = corrs[b]
                if len(pts_sar) < 10:
                    continue

                # Convert to feature-map indices (÷8)
                py1 = (pts_sar[:, 1] / 8).long()
                px1 = (pts_sar[:, 0] / 8).long()
                py2 = (pts_opt[:, 1] / 8).long()
                px2 = (pts_opt[:, 0] / 8).long()

                # Clamp to feature map bounds
                Hf, Wf = feats1.shape[-2:]
                py1 = py1.clamp(0, Hf - 1)
                px1 = px1.clamp(0, Wf - 1)
                py2 = py2.clamp(0, Hf - 1)
                px2 = px2.clamp(0, Wf - 1)

                m1 = feats1[b, :, py1, px1].permute(1, 0)
                m2 = feats2[b, :, py2, px2].permute(1, 0)
                h1 = hmap1[b, 0, py1, px1]
                h2 = hmap2[b, 0, py2, px2]

                # Losses
                loss_ds, conf = dual_softmax_loss(m1, m2)
                loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                loss_items += [loss_ds.unsqueeze(0), loss_kp.unsqueeze(0)]

                if _HAVE_ALIKE:
                    kp1 = kpts1[b]
                    kp2 = kpts2[b]
                    img1 = sars[b]
                    img2 = opts[b]
                    # alike_distill_loss runs on full image, not just correspondences
                    l_kp1, a1 = alike_distill_loss(kp1, img1)
                    l_kp2, a2 = alike_distill_loss(kp2, img2)
                    loss_items += [l_kp1.unsqueeze(0), l_kp2.unsqueeze(0)]
                    acc_kp_list.append((a1 + a2) / 2)

                acc_c_list.append(check_accuracy(m1, m2))

            if not loss_items:
                continue

            loss = torch.cat(loss_items).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()

            avg_acc_c = float(np.mean([v.cpu().numpy() if torch.is_tensor(v) else v for v in acc_c_list])) if acc_c_list else 0
            avg_acc_kp = float(np.mean([v.cpu().numpy() if torch.is_tensor(v) else v for v in acc_kp_list])) if acc_kp_list else 0

            total_loss += loss.item()
            total_acc_c += avg_acc_c
            total_acc_kp += avg_acc_kp
            n_batch += 1
            global_step += 1

            pbar.set_description(f"Loss: {loss.item():.4f}  acc_c: {avg_acc_c:.3f}  acc_kp: {avg_acc_kp:.3f}")

            # Log per step
            self.writer.add_scalar("Loss/train", loss.item(), global_step)
            self.writer.add_scalar("Accuracy/coarse", avg_acc_c, global_step)
            self.writer.add_scalar("Accuracy/keypoint", avg_acc_kp, global_step)
            self.history.append({"step": global_step, "loss": loss.item(), "acc_c": avg_acc_c, "acc_kp": avg_acc_kp})
            self._log(epoch, global_step, loss.item(), avg_acc_c, avg_acc_kp)

            if self.dry_run and global_step >= 10:
                break

        avg_loss = total_loss / max(n_batch, 1)
        avg_acc_c = float(total_acc_c / max(n_batch, 1))
        avg_acc_kp = float(total_acc_kp / max(n_batch, 1))
        return avg_loss, avg_acc_c, avg_acc_kp

    def train(self):
        print(f"[CrossModal] Starting training ...")
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            loss, acc_c, acc_kp = self.train_epoch(epoch)
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: loss={loss:.4f}  acc_c={acc_c:.4f}  acc_kp={acc_kp:.4f}  lr={current_lr:.6f}")

            self._save_latest()
            self._save_best(epoch, loss)
            self._plot_curves()
            self.scheduler.step()

            if self.dry_run and epoch >= 1:
                break

        self.writer.close()
        print(f"[CrossModal] Done. Best loss: {self.best_loss:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"D:\HanZhQ\PCIE715\Project_Trans\2350"))
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path (e.g. latest.pth)")
    p.add_argument("--ckpt-save-path", type=Path, default=Path("./checkpoints"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--lr-decay-steps", type=int, default=10, help="StepLR step size in epochs")
    p.add_argument("--resolution", type=str, default="512,512", help="Training resolution (W,H)")
    p.add_argument("--preprocess", choices=["raw", "grad", "clahe"], default="grad")
    p.add_argument("--device-num", type=str, default="0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Limit dataset size (0 = all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = CrossModalTrainer(args)
    trainer.train()
