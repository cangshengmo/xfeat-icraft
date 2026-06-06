"""Evaluate XFeat on SEN1-2 test set (4090 aligned SAR/OPT pairs)."""
from __future__ import annotations
import sys, csv, math
from pathlib import Path
import cv2, numpy as np, torch
from modules.xfeat import XFeat

REPO = Path(__file__).resolve().parent
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "weights" / "xfeat.pt")
DATA = Path(r"D:\HanZhQ\data\SEN1-2_256\test")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

# Load model
print(f"Loading weights: {WEIGHTS}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(WEIGHTS, map_location=dev)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    model = XFeat(weights=str(Path(__file__).resolve().parent / "weights" / "xfeat.pt"), top_k=4096)
    model.net.load_state_dict(ckpt["model_state"])
    print("Loaded best.pth (model_state)")
else:
    model = XFeat(weights=WEIGHTS, top_k=4096)
    print("Loaded xfeat.pt")

# Collect test pairs
sar_names = sorted({p.stem for p in (DATA/"sar").glob("*.*")})
opt_names = sorted({p.stem for p in (DATA/"opt").glob("*.*")})
ids = sorted(set(sar_names) & set(opt_names))[:LIMIT]
print(f"Evaluating {len(ids)} pairs from {DATA}")

rows = []
for sid in ids:
    sar = cv2.imread(str(DATA/"sar"/f"{sid}.bmp"), cv2.IMREAD_GRAYSCALE)
    opt = cv2.imread(str(DATA/"opt"/f"{sid}.bmp"), cv2.IMREAD_GRAYSCALE)

    # Preprocess (grad)
    for name, img in [("sar", sar), ("opt", opt)]:
        blur = cv2.GaussianBlur(img, (3,3), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        processed = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if name == "sar": sar = processed
        else: opt = processed

    # Extract features
    with torch.inference_mode():
        out1 = model.detectAndCompute(sar, top_k=4096)[0]
        out2 = model.detectAndCompute(opt, top_k=4096)[0]

    kp1, desc1 = out1["keypoints"].cpu(), out1["descriptors"].cpu()
    kp2, desc2 = out2["keypoints"].cpu(), out2["descriptors"].cpu()

    # Mutual NN matching
    sim = desc1 @ desc2.t()
    best2, best2_idx = sim.max(dim=1)
    best1 = sim.max(dim=0).indices
    keep = best1[best2_idx] == torch.arange(len(best2_idx))
    mkp1 = kp1[keep].numpy()
    mkp2 = kp2[best2_idx[keep]].numpy()
    n_matches = len(mkp1)

    # RANSAC with similarity model
    if n_matches >= 4:
        M, inliers = cv2.estimateAffinePartial2D(mkp1, mkp2, method=cv2.RANSAC,
            ransacReprojThreshold=6.0, maxIters=3000, confidence=0.995)
    else:
        M, inliers = None, None

    if M is not None:
        inlier_mask = inliers.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())
        # RMSE vs identity: predicted corners vs original corners
        corners = np.array([[0,0],[255,0],[255,255],[0,255]], dtype=np.float64)
        pred = (M[:,:2] @ corners.T + M[:,2:]).T
        rmse = float(np.sqrt(np.mean(np.sum(pred - corners)**2)))
    else:
        n_inliers, rmse = 0, math.nan

    rows.append({"id": sid, "matches": n_matches, "inliers": n_inliers, "rmse": rmse})

    status = f"{sid}: matches={n_matches}, inliers={n_inliers}"
    if not math.isnan(rmse): status += f", rmse={rmse:.2f}px"
    print(status)

# Summary
import numpy as np
rmses = np.array([r["rmse"] for r in rows if not math.isnan(r["rmse"])])
print(f"\n=== Summary ({len(rows)} pairs) ===")
if len(rmses):
    print(f"  ≤5px:  {(rmses<=5).sum()}/{len(rmses)}")
    print(f"  ≤10px: {(rmses<=10).sum()}/{len(rmses)}")
    print(f"  median RMSE: {np.median(rmses):.2f}px")
    print(f"  mean RMSE:   {np.mean(rmses):.2f}px")
