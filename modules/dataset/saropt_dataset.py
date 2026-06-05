"""
2350 SAR/OPT dataset utilities for XFeat fine-tuning.

Supports:
  1. Self-supervised warp training via SAROptAugmentationPipe
  2. Cross-modal pair training via SAROptPairDataset (for Phase 2)
"""

from __future__ import annotations

import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

from modules.dataset.augmentation import AugmentationPipe


class SAROptAugmentationPipe(AugmentationPipe):
    """
    AugmentationPipe variant that loads training images from the 2350
    SAR/OPT dataset instead of COCO.

    It collects images from both ``data_root / sar`` and ``data_root / opt``
    directories, resizes them to ``warp_resolution``, and feeds them through
    the standard random homography / TPS / photometric augmentation pipeline.

    Parameters
    ----------
    data_root : str or Path
        Path to the 2350 dataset root (must contain ``sar/`` and ``opt/`` subdirs).
    warp_resolution : tuple[int, int]
        (width, height) to resize images to before warping.
    max_num_imgs : int
        Maximum number of training images to load.
    **kwargs
        Extra arguments forwarded to ``AugmentationPipe``.
    """

    def __init__(
        self,
        data_root: str | Path,
        warp_resolution: tuple[int, int] = (800, 800),
        max_num_imgs: int = 500,
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.max_num_imgs = max_num_imgs
        self._user_warp_resolution = warp_resolution

        # Remove img_dir if provided — we handle it ourselves
        kwargs.pop("img_dir", None)
        # Set warp / out resolution
        kwargs.setdefault("out_resolution", warp_resolution)
        # We call super().__init__ with load_dataset=False and later call
        # load_imgs ourselves once self.dims is set.
        kwargs["load_dataset"] = False
        # Prevent automatic reload in forward() that would try to re-read from disk
        kwargs.setdefault("reload_step", 1_000_000)

        super().__init__(
            device=kwargs.pop("device", torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            **kwargs,
        )

        # Now manually load images
        self.load_imgs()

    def load_imgs(self):
        """Load images from 2350/sar and 2350/opt, resize to self.dims."""
        paths = []
        for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg"):
            paths.extend(glob.glob(str(self.data_root / "sar" / ext)))
            paths.extend(glob.glob(str(self.data_root / "opt" / ext)))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(
                f"No images found in {self.data_root / 'sar'} or {self.data_root / 'opt'}"
            )

        random.shuffle(paths)

        n_train = min(self.max_num_imgs, len(paths) - 10)
        n_test = min(10, len(paths) - n_train)

        train_paths = paths[:n_train]
        test_paths = paths[n_train : n_train + n_test]

        target_size = (self.dims[0], self.dims[1])  # (w, h)

        def _load_list(path_list, desc):
            out = []
            for p in tqdm.tqdm(path_list, desc=desc):
                im = cv2.imread(p)
                if im is None:
                    continue
                if im.shape[1] != target_size[0] or im.shape[0] != target_size[1]:
                    im = cv2.resize(im, target_size)
                out.append(np.copy(im))
            return out

        self.train = _load_list(train_paths, "[SAROpt] loading train")
        self.test = _load_list(test_paths, "[SAROpt] loading test")

        print(f"[SAROpt] Train: {len(self.train)}  Test: {len(self.test)}  images loaded.")


class SAROptPairDataset(torch.utils.data.Dataset):
    """
    Dataset that yields (sar_image, opt_image, homography) pairs from the 2350
    dataset for cross-modal supervised fine-tuning (Phase 2).

    Each item:
        sar_img   -> torch.Tensor (1, H_sar, W_sar)  grayscale
        opt_img   -> torch.Tensor (1, H_opt, W_opt)  grayscale
        H         -> torch.Tensor (3, 3)              homography from SAR → OPT
    """

    def __init__(
        self,
        data_root: str | Path,
        preprocess: str = "grad",
        sar_size: tuple[int, int] = (512, 512),
        opt_size: tuple[int, int] = (800, 800),
        limit: int = 0,
    ):
        self.data_root = Path(data_root)
        self.preprocess = preprocess
        self.sar_size = sar_size
        self.opt_size = opt_size

        self.labels = self._read_labels()
        if limit > 0:
            self.labels = self.labels[:limit]

    def _read_labels(self):
        rows = []
        path = self.data_root / "label.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append({"id": parts[0], "x": float(parts[1]), "y": float(parts[2])})
        return rows

    def _find_image(self, subdir, image_id):
        folder = self.data_root / subdir
        for ext in (".bmp", ".png", ".jpg", ".jpeg"):
            p = folder / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{folder / image_id}.* not found")

    def _preprocess(self, image: np.ndarray, mode: str) -> np.ndarray:
        if mode == "raw":
            return image
        if mode == "grad":
            blur = cv2.GaussianBlur(image, (3, 3), 0)
            gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        raise ValueError(f"Unknown preprocess: {mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels[idx]
        sar_path = self._find_image("sar", row["id"])
        opt_path = self._find_image("opt", row["id"])

        sar = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        opt = cv2.imread(str(opt_path), cv2.IMREAD_GRAYSCALE)

        sar = self._preprocess(sar, self.preprocess)
        opt = self._preprocess(opt, self.preprocess)

        # Build homography: translation-only from label.txt
        # SAR (512x512) is placed at (x, y) in OPT (800x800)
        H = np.eye(3, dtype=np.float32)
        H[0, 2] = row["x"]
        H[1, 2] = row["y"]

        sar_t = torch.from_numpy(sar).float().unsqueeze(0) / 255.0
        opt_t = torch.from_numpy(opt).float().unsqueeze(0) / 255.0
        H_t = torch.from_numpy(H)

        return sar_t, opt_t, H_t
