"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Keypoint extractor wrapper.  Tries to load ALIKE (deep model) first;
    falls back to OpenCV ORB when ALIKE is not available.
"""

import os
import sys
import warnings

import cv2
import numpy as np
import torch


# ── ALIKE (preferred) ───────────────────────────────────────────────
_HAVE_ALIKE = False
try:
    ALIKE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ALIKE"))
    sys.path.append(ALIKE_PATH)
    from alike import ALike  # noqa: F401

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _alike_model = ALike(
        **{
            "c1": 8, "c2": 16, "c3": 32, "c4": 64, "dim": 64,
            "single_head": True, "radius": 2,
            "model_path": os.path.join(ALIKE_PATH, "models", "alike-t.pth"),
        },
        device=dev, top_k=4096, scores_th=0.1, n_limit=8000,
    )
    _HAVE_ALIKE = True
except (ImportError, ModuleNotFoundError, FileNotFoundError):
    _alike_model = None


# ── ORB fallback ────────────────────────────────────────────────────
_orb_detector = cv2.ORB.create(nfeatures=8000, scoreType=cv2.ORB_FAST_SCORE)


def extract_kpts(img: np.ndarray) -> np.ndarray:
    """Return (N, 2) keypoint array in (x, y) format.

    ``img``: uint8 RGB or grayscale array, shape (H, W, 3) or (H, W).
    """
    if _HAVE_ALIKE and _alike_model is not None:
        pred = _alike_model(img, sub_pixel=True)
        return pred["keypoints"]

    # ORB fallback: handle any input format → uint8 grayscale
    img = np.asarray(img)
    if img.ndim == 2:
        # Already grayscale (H, W) — just ensure uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        gray = img
    else:
        # Color image (H, W, C) — convert to grayscale
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] >= 3 else img[:, :, 0]
    kps = _orb_detector.detect(gray, None)
    if not kps:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)  # (x, y)
    return pts


# ── Legacy aliases (for backward compatibility with losses.py) ───────
def extract_alike_kpts(img):
    return extract_kpts(img)


def detectAndCompute(img, top_k=4096):
    kpts = extract_kpts(img)
    scores = np.ones(len(kpts), dtype=np.float32)
    return (
        torch.tensor(kpts, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        torch.empty(0),
    )
