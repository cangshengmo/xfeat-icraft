"""
Phase 2: Cross-modal fine-tuning on 2350 data.
Loads the Phase 1 warp-trained checkpoint and fine-tunes on SAR/OPT pairs
using translation labels from label.txt.
"""

from __future__ import annotations

import argparse, csv, os, time, math
from pathlib import Path
import cv2, numpy as np, torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import tqdm

from modules.model import XFeatModel
from modules.training.losses import _HAVE_ALIKE, dual_softmax_loss, keypoint_loss
from modules.training.utils import check_accuracy, get_nb_trainable_params
if _HAVE_ALIKE:
    from modules.training.losses import alike_distill_loss


class Dataset2350Cross(Dataset):
    """2350 SAR/OPT pairs with translation labels for cross-modal fine-tuning."""

    def __init__(self, data_root, resolution=(512,512), preprocess="grad", limit=0, offset=0):
        self.data_root = Path(data_root)
        self.resolution = resolution  # (W, H)
        self.preprocess = preprocess

        labels = []
        for line in (data_root / "label.txt").read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 3:
                labels.append({"id": p[0], "x": float(p[1]), "y": float(p[2])})

        if limit > 0:
            labels = labels[offset:offset+limit]
        self.labels = labels
        print(f"[2350Cross] {len(self.labels)} pairs")

    def _find(self, subdir, image_id):
        folder = self.data_root / subdir
        for ext in (".bmp", ".png", ".jpg", ".jpeg"):
            p = folder / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{folder / image_id}.*")

    def _preprocess(self, img):
        if self.preprocess == "raw":
            return img
        blur = cv2.GaussianBlur(img, (3,3), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels[idx]
        sar = cv2.imread(str(self._find("sar", row["id"])), cv2.IMREAD_GRAYSCALE)
        opt = cv2.imread(str(self._find("opt", row["id"])), cv2.IMREAD_GRAYSCALE)
        sar = self._preprocess(sar)
        opt = self._preprocess(opt)

        # Resize to common resolution
        W, H = self.resolution
        sar = cv2.resize(sar, (W, H))
        opt = cv2.resize(opt, (W, H))

        # Scale translation: from (512x512→WxH), (800x800→WxH)
        sx = W / 512.0; sy = H / 512.0
        ox = W / 800.0; oy = H / 800.0
        tx = row["x"] * ox; ty = row["y"] * oy

        sar_t = torch.from_numpy(sar).float().unsqueeze(0) / 255.0
        opt_t = torch.from_numpy(opt).float().unsqueeze(0) / 255.0
        H_t = torch.tensor([[ox/sx, 0, tx], [0, oy/sy, ty], [0, 0, 1]], dtype=torch.float32)
        return sar_t, opt_t, H_t


def collate(batch):
    s, o, h = zip(*batch)
    return torch.stack(s), torch.stack(o), torch.stack(h)


def sample_corrs(H_batch, img_size, device, margin=8):
    """Sample dense grid with translation correspondences."""
    B = H_batch.shape[0]
    H_img, W_img = img_size
    h_g, w_g = H_img // 8, W_img // 8

    ys = torch.arange(margin, h_g - margin, device=device)
    xs = torch.arange(margin, w_g - margin, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1).float() * 8  # image-space

    results = []
    for b in range(B):
        H = H_batch[b]
        N = len(pts)
        ones = torch.ones(N, 1, device=device)
        sar_h = torch.cat([pts, ones], dim=-1)
        opt_h = (H @ sar_h.T).T
        pts_opt = opt_h[:, :2] / opt_h[:, 2:3]

        valid = (pts_opt[:, 0] >= 0) & (pts_opt[:, 0] < W_img) & \
                (pts_opt[:, 1] >= 0) & (pts_opt[:, 1] < H_img)
        if valid.sum() < 10:
            results.append((torch.zeros(0,2,device=device), torch.zeros(0,2,device=device)))
            continue
        # Subsample if too many
        sar_v, opt_v = pts[valid], pts_opt[valid]
        if len(sar_v) > 2048:
            perm = torch.randperm(len(sar_v), device=device)[:2048]
            sar_v, opt_v = sar_v[perm], opt_v[perm]
        results.append((sar_v, opt_v))
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(r"D:\HanZhQ\PCIE715\Project_Trans\2350"))
    p.add_argument("--resume", type=str, required=True, help="Phase 1 checkpoint (best.pth)")
    p.add_argument("--ckpt-save-path", type=Path, default=Path("./checkpoints"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resolution", type=str, default="512,512")
    p.add_argument("--preprocess", choices=["raw","grad"], default="grad")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase2] Device: {dev}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.ckpt_save_path / f"xfeat-2350-phase2-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Phase2] Output: {out_dir}")

    # Network
    net = XFeatModel().to(dev)
    ckpt = torch.load(args.resume, map_location=dev)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        net.load_state_dict(ckpt["model_state"])
        print(f"[Phase2] Loaded Phase 1 checkpoint (loss={ckpt.get('loss','?'):.4f})")
    else:
        net.load_state_dict(ckpt)
        print("[Phase2] Loaded plain state_dict")

    opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    res = tuple(int(v) for v in args.resolution.split(","))
    ds = Dataset2350Cross(args.data_root, resolution=res, preprocess=args.preprocess)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    img_size = (res[1], res[0])

    writer = SummaryWriter(str(out_dir / "tensorboard"))
    best_loss = float("inf")

    print(f"[Phase2] {len(ds)} pairs, {args.epochs} epochs, lr={args.lr}")
    get_nb_trainable_params(net)

    for epoch in range(args.epochs):
        net.train()
        losses, accs_c, accs_kp = [], [], []
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")

        for sars, opts, Hs in pbar:
            B = sars.shape[0]
            sars, opts, Hs = sars.to(dev), opts.to(dev), Hs.to(dev)
            corrs = sample_corrs(Hs, img_size, dev)

            feats1, kpts1, hmap1 = net(sars)
            feats2, kpts2, hmap2 = net(opts)

            items = []
            ac_list, ak_list = [], []

            for b in range(B):
                p1, p2 = corrs[b]
                if len(p1) < 10:
                    continue
                py = (p1[:, 1] / 8).long().clamp(0, feats1.shape[2]-1)
                px = (p1[:, 0] / 8).long().clamp(0, feats1.shape[3]-1)

                m1 = feats1[b, :, py, px].permute(1, 0)
                m2 = feats2[b, :, py, px].permute(1, 0)
                h1 = hmap1[b, 0, py, px]
                h2 = hmap2[b, 0, py, px]

                ld, conf = dual_softmax_loss(m1, m2)
                lk = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)
                items += [ld.unsqueeze(0), lk.unsqueeze(0)]

                if _HAVE_ALIKE:
                    l1, a1 = alike_distill_loss(kpts1[b], sars[b])
                    l2, a2 = alike_distill_loss(kpts2[b], opts[b])
                    items += [l1.unsqueeze(0), l2.unsqueeze(0)]
                    ak_list.append((a1 + a2) / 2)
                ac_list.append(check_accuracy(m1, m2))

            if not items:
                continue

            loss = torch.cat(items).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            avg_c = float(np.mean([v.cpu().numpy() if torch.is_tensor(v) else v for v in ac_list]))
            avg_k = float(np.mean([v.cpu().numpy() if torch.is_tensor(v) else v for v in ak_list])) if ak_list else 0
            losses.append(loss.item())
            accs_c.append(avg_c)
            accs_kp.append(avg_k)
            pbar.set_description(f"Loss: {loss.item():.4f}  acc_c: {avg_c:.3f}  acc_kp: {avg_k:.3f}")

        avg_loss = float(np.mean(losses))
        avg_acc_c = float(np.mean(accs_c))
        avg_acc_kp = float(np.mean(accs_kp))

        print(f"Epoch {epoch}: loss={avg_loss:.4f}  acc_c={avg_acc_c:.4f}  acc_kp={avg_acc_kp:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state": net.state_dict(), "loss": avg_loss}, out_dir / "best.pth")
            print(f"  ★ New best: {avg_loss:.4f}")

        if args.dry_run and epoch >= 1:
            break

    writer.close()
    print(f"[Phase2] Done. Best loss: {best_loss:.4f}")
