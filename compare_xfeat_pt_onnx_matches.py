"""
Visualize matching differences between PyTorch XFeat and ONNX XFeat.

The script uses the SAR/OPT dataset under:
    D:\\HanZhQ\\PCIE715\\Project_Trans\\2350

For each image id it runs:
  1. PyTorch XFeat detectAndCompute
  2. ONNX XFeat frontend + the same host-side sparse post-processing
  3. mutual nearest-neighbor descriptor matching
  4. similarity/affine/homography RANSAC

It saves a stacked visualization: PyTorch result on top, ONNX result below.
Green lines are RANSAC inliers, red lines are outliers. Blue polygon is ground
truth SAR placement, yellow polygon is the estimated placement.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from modules.interpolator import InterpolateSparse2d
from modules.xfeat import XFeat
from xfeat_sar_opt_eval import (
    LabelRow,
    affine_stats,
    corner_rmse,
    estimate_transform,
    find_image,
    make_gt_aug_to_opt,
    mutual_nn_matches,
    parse_float_list,
    preprocess_gray,
    read_gray,
    read_labels,
    selected_labels,
    synthetic_sar_view,
    transform_points,
)


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\HanZhQ\PCIE715\Project_Trans\2350")
DEFAULT_ONNX_512 = REPO_DIR / "outputs" / "xfeat_onnx" / "xfeat_frontend_512x512.onnx"
DEFAULT_ONNX_800 = REPO_DIR / "outputs" / "xfeat_onnx" / "xfeat_frontend_800x800.onnx"
DEFAULT_WEIGHTS = REPO_DIR / "weights" / "xfeat.pt"


@dataclass
class MatchResult:
    backend: str
    match_count: int
    inlier_count: int
    inlier_ratio: float
    accepted: bool
    rmse: float
    matrix: np.ndarray | None
    inliers: np.ndarray | None
    opt_pts: np.ndarray
    sar_pts: np.ndarray
    tx: float
    ty: float
    estimated_scale: float
    estimated_angle_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PT and ONNX XFeat matching on SAR/OPT data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--onnx-512", type=Path, default=DEFAULT_ONNX_512, help="ONNX model for 512x512 SAR images.")
    parser.add_argument("--onnx-800", type=Path, default=DEFAULT_ONNX_800, help="ONNX model for 800x800 OPT images.")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "outputs" / "xfeat_pt_vs_onnx_matches")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=3072)
    parser.add_argument("--detection-threshold", type=float, default=0.05)
    parser.add_argument("--min-cossim", type=float, default=-1.0)
    parser.add_argument("--ransac-thr", type=float, default=6.0)
    parser.add_argument("--model", choices=["similarity", "affine", "homography"], default="similarity")
    parser.add_argument("--preprocess", choices=["raw", "clahe", "grad", "canny"], default="grad")
    parser.add_argument("--sar-blur", type=int, default=3)
    parser.add_argument("--angles", default="0")
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--min-accept-inliers", type=int, default=30)
    parser.add_argument("--min-accept-ratio", type=float, default=0.04)
    parser.add_argument("--max-draw", type=int, default=250)
    parser.add_argument("--force-cpu", action="store_true", help="Force PyTorch XFeat to CPU.")
    return parser.parse_args()


def normalize_onnx_output(name: str, array: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    out_h, out_w = h // 8, w // 8
    channels = {"dense_descriptors": 64, "keypoint_logits": 65, "reliability": 1}[name]
    nchw = (1, channels, out_h, out_w)
    nhwc = (1, out_h, out_w, channels)
    if array.shape == nchw:
        return array.astype(np.float32, copy=False)
    if array.shape == nhwc:
        return np.transpose(array, (0, 3, 1, 2)).astype(np.float32, copy=False)
    raise ValueError(f"{name} shape {array.shape} does not match expected {nchw} or {nhwc}")


def get_kpts_heatmap(kpts: torch.Tensor, softmax_temp: float = 1.0) -> torch.Tensor:
    scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
    batch, _, height, width = scores.shape
    heatmap = scores.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
    heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(batch, 1, height * 8, width * 8)
    return heatmap


def nms(heatmap: torch.Tensor, threshold: float, kernel_size: int = 5) -> torch.Tensor:
    batch, _, height, width = heatmap.shape
    pad = kernel_size // 2
    local_max = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(heatmap)
    pos = (heatmap == local_max) & (heatmap > threshold)
    pos_batched = [item.nonzero()[..., 1:].flip(-1) for item in pos]
    pad_val = max([len(item) for item in pos_batched] + [1])
    points = torch.zeros((batch, pad_val, 2), dtype=torch.long, device=heatmap.device)
    for index, item in enumerate(pos_batched):
        points[index, : len(item), :] = item
    return points


@torch.inference_mode()
def decode_onnx_sparse(
    dense_descriptors: np.ndarray,
    keypoint_logits: np.ndarray,
    reliability: np.ndarray,
    top_k: int,
    detection_threshold: float,
) -> dict[str, torch.Tensor]:
    m1 = torch.from_numpy(dense_descriptors).float()
    k1 = torch.from_numpy(keypoint_logits).float()
    h1 = torch.from_numpy(reliability).float()

    m1 = F.normalize(m1, dim=1)
    k1h = get_kpts_heatmap(k1)
    mkpts = nms(k1h, threshold=detection_threshold, kernel_size=5)

    nearest = InterpolateSparse2d("nearest")
    bilinear = InterpolateSparse2d("bilinear")
    scores = (nearest(k1h, mkpts, k1h.shape[-2], k1h.shape[-1]) * bilinear(h1, mkpts, k1h.shape[-2], k1h.shape[-1])).squeeze(-1)
    scores[torch.all(mkpts == 0, dim=-1)] = -1

    idxs = torch.argsort(-scores)
    mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
    mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
    mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
    scores = torch.gather(scores, -1, idxs)[:, :top_k]

    descriptors = InterpolateSparse2d("bicubic")(m1, mkpts, H=k1h.shape[-2], W=k1h.shape[-1])
    descriptors = F.normalize(descriptors, dim=-1)

    valid = scores[0] > 0
    return {
        "keypoints": mkpts[0][valid].float(),
        "scores": scores[0][valid].float(),
        "descriptors": descriptors[0][valid].float(),
    }


class OnnxXFeatFrontend:
    def __init__(self, onnx_512: Path, onnx_800: Path):
        if not onnx_512.exists():
            raise FileNotFoundError(f"ONNX 512 model not found: {onnx_512}")
        if not onnx_800.exists():
            raise FileNotFoundError(f"ONNX 800 model not found: {onnx_800}")
        self.sessions = {
            (512, 512): ort.InferenceSession(str(onnx_512), providers=["CPUExecutionProvider"]),
            (800, 800): ort.InferenceSession(str(onnx_800), providers=["CPUExecutionProvider"]),
        }

    def detect_and_compute(self, image: np.ndarray, top_k: int, detection_threshold: float) -> dict[str, torch.Tensor]:
        image_shape = image.shape[:2]
        session = self.sessions.get(image_shape)
        if session is None:
            raise ValueError(f"No ONNX session for image shape {image_shape}. Expected 512x512 or 800x800.")
        tensor = image.astype(np.float32)[None, None, :, :]
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        dense = normalize_onnx_output("dense_descriptors", outputs[0], image_shape)
        logits = normalize_onnx_output("keypoint_logits", outputs[1], image_shape)
        reliability = normalize_onnx_output("reliability", outputs[2], image_shape)
        return decode_onnx_sparse(dense, logits, reliability, top_k, detection_threshold)


def load_pt_xfeat(weights: Path, top_k: int, force_cpu: bool) -> XFeat:
    if force_cpu:
        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            return XFeat(weights=str(weights), top_k=top_k)
        finally:
            torch.cuda.is_available = original
    return XFeat(weights=str(weights), top_k=top_k)


@torch.inference_mode()
def extract_pt_features(xfeat: XFeat, image: np.ndarray, top_k: int, detection_threshold: float) -> dict[str, torch.Tensor]:
    return xfeat.detectAndCompute(image, top_k=top_k, detection_threshold=detection_threshold)[0]


def run_matching(
    backend: str,
    opt_feat: dict[str, torch.Tensor],
    sar_feat: dict[str, torch.Tensor],
    args: argparse.Namespace,
    gt_aug_to_opt: np.ndarray,
    sar_shape: tuple[int, int],
) -> MatchResult:
    sar_idx, opt_idx, _scores = mutual_nn_matches(
        sar_feat["descriptors"],
        opt_feat["descriptors"],
        args.min_cossim,
    )
    sar_pts = sar_feat["keypoints"][sar_idx].cpu().numpy().astype(np.float32)
    opt_pts = opt_feat["keypoints"][opt_idx].cpu().numpy().astype(np.float32)
    matrix, inliers = estimate_transform(sar_pts, opt_pts, args.model, args.ransac_thr)

    match_count = int(len(sar_pts))
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    inlier_ratio = float(inlier_count / match_count) if match_count else 0.0
    accepted = inlier_count >= args.min_accept_inliers and inlier_ratio >= args.min_accept_ratio
    rmse = corner_rmse(matrix, sar_shape[1], sar_shape[0], gt_aug_to_opt)
    tx, ty, est_scale, est_angle = affine_stats(matrix)

    return MatchResult(
        backend=backend,
        match_count=match_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        accepted=accepted,
        rmse=rmse,
        matrix=matrix,
        inliers=inliers,
        opt_pts=opt_pts,
        sar_pts=sar_pts,
        tx=tx,
        ty=ty,
        estimated_scale=est_scale,
        estimated_angle_deg=est_angle,
    )


def draw_panel(
    opt_img: np.ndarray,
    sar_img: np.ndarray,
    result: MatchResult,
    gt_aug_to_opt: np.ndarray,
    max_draw: int,
) -> np.ndarray:
    opt_vis = cv2.cvtColor(opt_img, cv2.COLOR_GRAY2BGR)
    sar_vis = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    gap = 24
    height = max(opt_vis.shape[0], sar_vis.shape[0])
    width = opt_vis.shape[1] + gap + sar_vis.shape[1]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[: opt_vis.shape[0], : opt_vis.shape[1]] = opt_vis
    canvas[: sar_vis.shape[0], opt_vis.shape[1] + gap :] = sar_vis

    mask = np.zeros((len(result.sar_pts),), dtype=bool)
    if result.inliers is not None:
        mask = result.inliers.reshape(-1).astype(bool)

    order = np.argsort(mask.astype(np.int32))
    if len(order) > max_draw:
        outlier_idx = order[~mask[order]][: max_draw // 2]
        inlier_idx = order[mask[order]][: max_draw - len(outlier_idx)]
        order = np.concatenate([outlier_idx, inlier_idx])

    x_offset = opt_vis.shape[1] + gap
    for idx in order:
        pt_opt = tuple(np.round(result.opt_pts[idx]).astype(int))
        pt_sar = tuple(np.round(result.sar_pts[idx] + np.array([x_offset, 0])).astype(int))
        color = (0, 220, 0) if mask[idx] else (0, 0, 255)
        cv2.line(canvas, pt_opt, pt_sar, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt_opt, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt_sar, 2, color, -1, cv2.LINE_AA)

    sar_corners = np.array(
        [[0, 0], [sar_img.shape[1] - 1, 0], [sar_img.shape[1] - 1, sar_img.shape[0] - 1], [0, sar_img.shape[0] - 1]],
        dtype=np.float64,
    )
    gt_poly = np.round(transform_points(sar_corners, gt_aug_to_opt)).astype(np.int32)
    cv2.polylines(canvas, [gt_poly], True, (255, 180, 0), 2, cv2.LINE_AA)
    if result.matrix is not None:
        pred_poly = np.round(transform_points(sar_corners, result.matrix)).astype(np.int32)
        cv2.polylines(canvas, [pred_poly], True, (0, 255, 255), 2, cv2.LINE_AA)

    title = (
        f"{result.backend} matches={result.match_count} inliers={result.inlier_count} "
        f"ratio={result.inlier_ratio:.3f} rmse={result.rmse:.2f}px accepted={int(result.accepted)}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (245, 245, 245), -1)
    cv2.putText(canvas, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def save_stacked_visual(
    opt_img: np.ndarray,
    sar_img: np.ndarray,
    pt_result: MatchResult,
    onnx_result: MatchResult,
    gt_aug_to_opt: np.ndarray,
    save_path: Path,
    max_draw: int,
) -> None:
    pt_panel = draw_panel(opt_img, sar_img, pt_result, gt_aug_to_opt, max_draw)
    onnx_panel = draw_panel(opt_img, sar_img, onnx_result, gt_aug_to_opt, max_draw)
    width = max(pt_panel.shape[1], onnx_panel.shape[1])

    def pad(panel: np.ndarray) -> np.ndarray:
        if panel.shape[1] == width:
            return panel
        out = np.full((panel.shape[0], width, 3), 245, dtype=np.uint8)
        out[:, : panel.shape[1]] = panel
        return out

    stacked = np.vstack([pad(pt_panel), pad(onnx_panel)])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), stacked)


def row_for(label: LabelRow, angle: float, scale: float, result: MatchResult, draw_path: Path) -> dict[str, object]:
    return {
        "image_id": label.image_id,
        "angle_deg": angle,
        "synthetic_scale": scale,
        "backend": result.backend,
        "matches": result.match_count,
        "inliers": result.inlier_count,
        "inlier_ratio": result.inlier_ratio,
        "accepted": int(result.accepted),
        "corner_rmse": result.rmse,
        "tx": result.tx,
        "ty": result.ty,
        "estimated_scale": result.estimated_scale,
        "estimated_angle_deg": result.estimated_angle_deg,
        "draw_path": str(draw_path),
    }


def write_summary(rows: list[dict[str, object]], csv_path: Path) -> None:
    fieldnames = [
        "image_id",
        "angle_deg",
        "synthetic_scale",
        "backend",
        "matches",
        "inliers",
        "inlier_ratio",
        "accepted",
        "corner_rmse",
        "tx",
        "ty",
        "estimated_scale",
        "estimated_angle_deg",
        "draw_path",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_backend_summary(rows: list[dict[str, object]], backend: str) -> None:
    selected = [row for row in rows if row["backend"] == backend]
    rmse = np.array([float(row["corner_rmse"]) for row in selected if not math.isnan(float(row["corner_rmse"]))])
    if len(rmse) == 0:
        print(f"{backend}: cases={len(selected)} valid=0")
        return
    accepted_rmse = np.array(
        [
            float(row["corner_rmse"])
            for row in selected
            if int(row["accepted"]) == 1 and not math.isnan(float(row["corner_rmse"]))
        ]
    )
    print(
        f"{backend}: cases={len(selected)} valid={len(rmse)} "
        f"<=5px={(rmse <= 5).sum()} <=10px={(rmse <= 10).sum()} "
        f"median={np.median(rmse):.2f}px mean={np.mean(rmse):.2f}px accepted={len(accepted_rmse)}"
    )
    if len(accepted_rmse):
        print(
            f"{backend} accepted: <=5px={(accepted_rmse <= 5).sum()} "
            f"<=10px={(accepted_rmse <= 10).sum()} "
            f"median={np.median(accepted_rmse):.2f}px mean={np.mean(accepted_rmse):.2f}px"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = selected_labels(read_labels(args.data_root / "label.txt"), args.ids, args.limit)
    angles = parse_float_list(args.angles)
    scales = parse_float_list(args.scales)
    pt_xfeat = load_pt_xfeat(args.weights, args.top_k, args.force_cpu)
    onnx_xfeat = OnnxXFeatFrontend(args.onnx_512, args.onnx_800)

    rows: list[dict[str, object]] = []
    for label in labels:
        opt_raw = read_gray(find_image(args.data_root, "opt", label.image_id))
        sar_raw = read_gray(find_image(args.data_root, "sar", label.image_id))
        opt = preprocess_gray(opt_raw, args.preprocess, is_sar=False, sar_blur=args.sar_blur)

        for angle in angles:
            for scale in scales:
                sar_aug_raw, aug_to_src = synthetic_sar_view(sar_raw, angle, scale)
                sar = preprocess_gray(sar_aug_raw, args.preprocess, is_sar=True, sar_blur=args.sar_blur)
                gt_aug_to_opt = make_gt_aug_to_opt(label.x, label.y, aug_to_src)

                pt_opt_feat = extract_pt_features(pt_xfeat, opt, args.top_k, args.detection_threshold)
                pt_sar_feat = extract_pt_features(pt_xfeat, sar, args.top_k, args.detection_threshold)
                onnx_opt_feat = onnx_xfeat.detect_and_compute(opt, args.top_k, args.detection_threshold)
                onnx_sar_feat = onnx_xfeat.detect_and_compute(sar, args.top_k, args.detection_threshold)

                pt_result = run_matching("PyTorch", pt_opt_feat, pt_sar_feat, args, gt_aug_to_opt, sar.shape[:2])
                onnx_result = run_matching("ONNX", onnx_opt_feat, onnx_sar_feat, args, gt_aug_to_opt, sar.shape[:2])

                stem = f"{label.image_id}_a{angle:g}_s{scale:g}_{args.preprocess}_{args.model}.jpg"
                draw_path = args.output_dir / "matches" / stem
                save_stacked_visual(opt, sar, pt_result, onnx_result, gt_aug_to_opt, draw_path, args.max_draw)

                rows.append(row_for(label, angle, scale, pt_result, draw_path))
                rows.append(row_for(label, angle, scale, onnx_result, draw_path))
                print(
                    f"{label.image_id} angle={angle:g} scale={scale:g}: "
                    f"PT matches={pt_result.match_count} inliers={pt_result.inlier_count} rmse={pt_result.rmse:.2f}px | "
                    f"ONNX matches={onnx_result.match_count} inliers={onnx_result.inlier_count} rmse={onnx_result.rmse:.2f}px"
                )

    csv_path = args.output_dir / "xfeat_pt_vs_onnx_summary.csv"
    write_summary(rows, csv_path)
    print("")
    print(f"CSV: {csv_path}")
    print_backend_summary(rows, "PyTorch")
    print_backend_summary(rows, "ONNX")


if __name__ == "__main__":
    main()
