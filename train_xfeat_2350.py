"""
Fine-tune XFeat on the 2350 SAR/OPT dataset using self-supervised warp training.

The training output directory is auto-named as:
    {ckpt-save-path}/{model}-{dataset}-{timestamp}/

Inside:
    best.pth           — model with the lowest loss so far
    latest.pth         — most recent model (for resuming on interrupt)
    {name}_step_{N}.pth — periodic checkpoints
    training_log.csv   — step-by-step loss / accuracy / lr
    curves.png         — loss & accuracy curves
    tensorboard/       — TensorBoard event files
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import tqdm

from modules.model import XFeatModel
from modules.training.losses import (
    _HAVE_ALIKE,
    coordinate_classification_loss,
    dual_softmax_loss,
    keypoint_loss,
)
from modules.training.utils import check_accuracy, get_corresponding_pts, get_nb_trainable_params, make_batch

if _HAVE_ALIKE:
    from modules.training.losses import alike_distill_loss
else:
    print("[Trainer2350] ALIKE not available — skipping keypoint distillation loss.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune XFeat on 2350 SAR/OPT data.")
    p.add_argument("--data-root", type=Path, default=Path(r"D:\HanZhQ\PCIE715\Project_Trans\2350"),
                   help="2350 dataset root with sar/ opt/ label.txt")
    p.add_argument("--weights", type=str, default=None,
                   help="Initial weights path. Default: weights/xfeat.pt")
    p.add_argument("--ckpt-save-path", type=Path, default=Path("./checkpoints"),
                   help="Parent directory for timestamped run folders")
    p.add_argument("--name", type=str, default="xfeat",
                   help="Model name used in output folder: {name}-{dataset}-{timestamp}")
    p.add_argument("--dataset", type=str, default="2350",
                   help="Dataset name used in output folder")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p.add_argument("--n-steps", type=int, default=20_000, help="Number of training steps")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma-steplr", type=float, default=0.5, help="StepLR gamma")
    p.add_argument("--training-res", type=str, default="800,608",
                   help="Training resolution (width,height)")
    p.add_argument("--device-num", type=str, default="0", help="CUDA device number")
    p.add_argument("--dry-run", action="store_true", help="Single mini-batch sanity check")
    p.add_argument("--save-ckpt-every", type=int, default=1000,
                   help="Save periodic checkpoint every N steps")
    p.add_argument("--max-imgs", type=int, default=500,
                   help="Max training images to load from 2350")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a latest.pth or step checkpoint to resume from")
    return p.parse_args()


class Trainer2350:
    """Fine-tune XFeat on 2350 data via self-supervised warp augmentation."""

    def __init__(self, args: argparse.Namespace):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        self.args = args
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer2350] Device: {self.dev}")

        # ── Output directory (timestamped) ──────────────────────────
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.name}-{args.dataset}-{ts}"
        self.out_dir: Path = args.ckpt_save_path / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Trainer2350] Output: {self.out_dir}")

        # ── Network ─────────────────────────────────────────────────
        self.net = XFeatModel().to(self.dev)
        start_step = 0

        if args.resume:
            # Resume from a checkpoint
            ckpt = torch.load(args.resume, map_location=self.dev)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                self.net.load_state_dict(ckpt["model_state"])
                start_step = ckpt.get("step", 0)
                print(f"[Trainer2350] Resumed from step {start_step}: {args.resume}")
            else:
                self.net.load_state_dict(ckpt)
                print(f"[Trainer2350] Loaded weights (no step info): {args.resume}")
        else:
            weights_path = args.weights or (Path(__file__).resolve().parent / "weights" / "xfeat.pt")
            if weights_path.exists():
                print(f"[Trainer2350] Loading pretrained weights from: {weights_path}")
                self.net.load_state_dict(torch.load(str(weights_path), map_location=self.dev))
            else:
                print(f"[Trainer2350] WARNING: weights not found — training from scratch")

        self.start_step = start_step

        # ── Optimiser ───────────────────────────────────────────────
        self.opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), lr=args.lr
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=args.gamma_steplr)

        # ── Augmentation pipe ───────────────────────────────────────
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

        # ── Logging infrastructure ──────────────────────────────────
        self.steps = args.n_steps
        self.save_ckpt_every = args.save_ckpt_every
        self.dry_run = args.dry_run

        # TensorBoard
        self.writer = SummaryWriter(str(self.out_dir / "tensorboard"))

        # CSV log
        self.csv_path = self.out_dir / "training_log.csv"
        self._init_csv()

        # History for plotting
        self.history: list[dict] = []

        # Best-loss tracker
        self.best_loss = float("inf")

        print(f"[Trainer2350] Total steps: {self.steps}  |  Log: {self.csv_path}")
        get_nb_trainable_params(self.net)

    # ── helpers ─────────────────────────────────────────────────────

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", "acc_c", "acc_f", "acc_kp", "lr"])

    def _append_csv(self, step, loss, acc_c, acc_f, acc_kp, lr):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, f"{loss:.6f}", f"{acc_c:.4f}", f"{acc_f:.4f}", f"{acc_kp:.4f}", f"{lr:.6f}"])

    def _save_latest(self, step):
        """Overwrite latest.pth so training can be resumed after interrupt."""
        path = self.out_dir / "latest.pth"
        torch.save({
            "step": step,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "args": self.args,
        }, str(path))

    def _save_best(self, step, loss):
        """Overwrite best.pth when loss improves."""
        if loss >= self.best_loss:
            return
        self.best_loss = loss
        path = self.out_dir / "best.pth"
        torch.save({
            "step": step,
            "model_state": self.net.state_dict(),
            "loss": loss,
        }, str(path))
        print(f"\n[Trainer2350] ★ New best loss {loss:.4f} — saved best.pth")

    def _save_periodic(self, step):
        path = self.out_dir / f"xfeat_2350_step_{step}.pth"
        torch.save(self.net.state_dict(), str(path))
        print(f"\n[Trainer2350] Periodic checkpoint saved: {path.name}")

    def _plot_curves(self):
        """Generate loss & accuracy curves from self.history and save as PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return  # matplotlib not available

        if len(self.history) < 3:
            return

        steps = [h["step"] for h in self.history]
        losses = [h["loss"] for h in self.history]
        acc_c = [h["acc_c"] for h in self.history]
        acc_f = [h["acc_f"] for h in self.history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(steps, losses, label="Loss", color="tab:red")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(steps, acc_c, label="acc_c (coarse match)", color="tab:blue")
        axes[1].plot(steps, acc_f, label="acc_f (fine coord)", color="tab:orange")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Matching Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(self.out_dir / "curves.png"), dpi=150)
        plt.close(fig)
        print(f"[Trainer2350] Curves saved to curves.png")

    # ── main training loop ──────────────────────────────────────────

    def train(self):
        self.net.train()
        difficulty = 0.10

        print(f"[Trainer2350] Starting training ...")
        pbar = tqdm.tqdm(total=self.steps, initial=self.start_step)

        for i in range(self.start_step, self.steps):
            # ── Grab synthetic warp batch ────────────────────────────
            p1, p2, H1, H2 = make_batch(self.augmentor, difficulty)

            # ── Convert to grayscale ─────────────────────────────────
            p1 = p1.mean(1, keepdim=True)
            p2 = p2.mean(1, keepdim=True)

            # ── Get ground-truth correspondences from warp params ────
            h_coarse, w_coarse = p1.shape[-2] // 8, p1.shape[-1] // 8
            _, positives = get_corresponding_pts(
                p1, p2, H1, H2, self.augmentor, h_coarse, w_coarse
            )

            # ── Skip corrupted batches ───────────────────────────────
            if any(len(p) < 30 for p in positives):
                continue

            # ── Forward ──────────────────────────────────────────────
            feats1, kpts1, hmap1 = self.net(p1)
            feats2, kpts2, hmap2 = self.net(p2)

            loss_items = []
            total_acc_coarse = 0.0
            total_acc_fine = 0.0
            total_acc_pos = 0.0
            nb = len(positives)

            for b in range(nb):
                pts1, pts2 = positives[b][:, :2], positives[b][:, 2:]

                m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
                m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)
                h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]
                coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                loss_ds, conf = dual_softmax_loss(m1, m2)
                loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                if _HAVE_ALIKE:
                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                    acc_pos = (acc_pos1 + acc_pos2) / 2.0
                else:
                    loss_kp_pos = torch.tensor(0.0, device=self.dev)
                    acc_pos = 1.0

                loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                loss_items += [
                    loss_ds.unsqueeze(0),
                    loss_coords.unsqueeze(0),
                    loss_kp.unsqueeze(0),
                    loss_kp_pos.unsqueeze(0),
                ]

                total_acc_coarse += check_accuracy(m1, m2)
                total_acc_fine += acc_coords
                total_acc_pos += acc_pos

            # ── Backward ─────────────────────────────────────────────
            loss = torch.cat(loss_items, -1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()
            self.scheduler.step()

            # ── Aggregate metrics ────────────────────────────────────
            current_lr = self.scheduler.get_last_lr()[0]
            avg_acc_coarse = total_acc_coarse / nb
            avg_acc_fine = total_acc_fine / nb
            avg_acc_pos = total_acc_pos / nb
            loss_val = loss.item()

            # ── Logging ──────────────────────────────────────────────
            pbar.set_description(
                f"Loss: {loss_val:.4f}  acc_c: {avg_acc_coarse:.3f}  "
                f"acc_f: {avg_acc_fine:.3f}  acc_kp: {avg_acc_pos:.3f}"
            )
            pbar.update(1)

            # TensorBoard
            global_step = i + 1
            self.writer.add_scalar("Loss/total", loss_val, global_step)
            self.writer.add_scalar("Accuracy/coarse", avg_acc_coarse, global_step)
            self.writer.add_scalar("Accuracy/fine", avg_acc_fine, global_step)
            self.writer.add_scalar("Accuracy/keypoint_position", avg_acc_pos, global_step)
            self.writer.add_scalar("lr", current_lr, global_step)

            # CSV & history (sample once per save interval to keep file small)
            record = {
                "step": global_step,
                "loss": float(loss_val),
                "acc_c": float(avg_acc_coarse),
                "acc_f": float(avg_acc_fine),
                "acc_kp": float(avg_acc_pos),
            }
            self.history.append(record)
            self._append_csv(global_step, loss_val, avg_acc_coarse, avg_acc_fine, avg_acc_pos, current_lr)

            # ── Checkpoint management ────────────────────────────────
            if not self.dry_run:
                # Always update latest.pth (for resume)
                self._save_latest(global_step)

                # Track best (lowest loss)
                self._save_best(global_step, loss_val)

                # Periodic checkpoint
                if global_step % self.save_ckpt_every == 0:
                    self._save_periodic(global_step)
                    self._plot_curves()  # update curves each periodic save

            if self.dry_run and i - self.start_step >= 2:
                print("[Trainer2350] Dry-run complete (3 steps after start).")
                break

        pbar.close()
        self._plot_curves()
        self.writer.close()
        print(f"[Trainer2350] Training complete.  Best loss: {self.best_loss:.4f}")
        print(f"[Trainer2350] All artifacts in: {self.out_dir}")


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer2350(args)
    trainer.train()
