"""
使用 XFeat 对光学/SAR 图像做最小特征匹配验证。

核心思路：
1. 芯片侧未来只部署 XFeat 前向特征提取网络，输出描述子、关键点 logits、可靠性图。
2. 当前脚本中的互最近邻匹配、RANSAC、画图均为主机侧后处理，不强行放进 ICRAFT。
3. label.txt 的每行格式为：图像 id、SAR 左上角 x、SAR 左上角 y、无意义字段。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\HanZhQ\PCIE715\Project_Trans\2350")
DEFAULT_XFEAT_ROOT = REPO_DIR


@dataclass(frozen=True)
class LabelRow:
    image_id: str
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XFeat 光学/SAR 特征匹配最小验证，输出 CSV 和绿线内点/红线外点连线图。"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="包含 opt、sar、label.txt 的数据目录。")
    parser.add_argument("--xfeat-root", type=Path, default=DEFAULT_XFEAT_ROOT, help="XFeat 源码目录。")
    parser.add_argument("--weights", type=Path, default=None, help="XFeat 权重路径，默认使用 xfeat-root/weights/xfeat.pt。")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "outputs" / "xfeat_sar_opt", help="结果输出目录。")
    parser.add_argument("--limit", type=int, default=10, help="最多评估多少组图像。")
    parser.add_argument("--ids", nargs="*", default=None, help="只评估指定图像 id，例如 100001 100002。")
    parser.add_argument("--top-k", type=int, default=1024, help="每幅图最多保留的 XFeat 关键点数。")
    parser.add_argument("--min-cossim", type=float, default=-1.0, help="互最近邻后的最小余弦相似度，-1 表示不过滤。")
    parser.add_argument("--ransac-thr", type=float, default=6.0, help="RANSAC 重投影阈值，单位为像素。")
    parser.add_argument(
        "--model",
        choices=["similarity", "affine", "homography"],
        default="similarity",
        help="几何模型：similarity=旋转+尺度+平移，affine=完整仿射，homography=单应。",
    )
    parser.add_argument(
        "--preprocess",
        choices=["raw", "clahe", "grad", "canny"],
        default="clahe",
        help="输入 XFeat 前的灰度预处理。clahe 通常比 raw 更适合跨模态初测。",
    )
    parser.add_argument("--sar-blur", type=int, default=3, help="SAR 预处理前的中值滤波核大小，0/1 表示关闭。")
    parser.add_argument("--angles", default="0", help="对 SAR 图像额外施加的合成旋转角度，逗号分隔。")
    parser.add_argument("--scales", default="1.0", help="对 SAR 图像额外施加的合成尺度，逗号分隔。")
    parser.add_argument("--min-accept-inliers", type=int, default=0, help="质量门控：RANSAC 内点数至少达到该值才标记为 accepted。")
    parser.add_argument("--min-accept-ratio", type=float, default=0.0, help="质量门控：RANSAC 内点比例至少达到该值才标记为 accepted。")
    parser.add_argument(
        "--rmse-transform",
        type=str,
        default="log",
        choices=["none", "cap", "log"],
        help="RMSE 鲁棒聚合方式："
             "none=原始值；"
             "cap=硬截断（结合 --rmse-threshold）；"
             "log=对数压缩（推荐，默认），对超出阈值的部分做对数平滑。",
    )
    parser.add_argument("--rmse-threshold", type=float, default=10.0, help="RMSE 变换阈值（像素），仅对超出该值的部分做处理。")
    parser.add_argument("--max-draw", type=int, default=250, help="每张连线图最多绘制多少条匹配线。")
    parser.add_argument("--no-images", action="store_true", help="只输出 CSV，不保存连线图。")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用 CPU，默认优先使用 CUDA。")
    return parser.parse_args()


def read_labels(label_path: Path) -> list[LabelRow]:
    rows: list[LabelRow] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append(LabelRow(parts[0], float(parts[1]), float(parts[2])))
    return rows


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    return values or [0.0]


def find_image(root: Path, subdir: str, image_id: str) -> Path:
    folder = root / subdir
    for ext in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = folder / f"{image_id}{ext}"
        if path.exists():
            return path
    matches = sorted(folder.glob(f"{image_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"找不到图像：{folder / image_id}.*")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return image


def preprocess_gray(image: np.ndarray, mode: str, is_sar: bool, sar_blur: int) -> np.ndarray:
    out = image
    if is_sar and sar_blur and sar_blur > 1:
        kernel = sar_blur if sar_blur % 2 == 1 else sar_blur + 1
        out = cv2.medianBlur(out, kernel)

    if mode == "raw":
        return out

    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(out)

    if mode == "grad":
        blur = cv2.GaussianBlur(out, (3, 3), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if mode == "canny":
        blur = cv2.GaussianBlur(out, (3, 3), 0)
        edge = cv2.Canny(blur, 50, 150)
        return cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)

    raise ValueError(f"未知预处理模式：{mode}")


def synthetic_sar_view(image: np.ndarray, angle_deg: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """返回合成 SAR 图和从合成图坐标到原 SAR 坐标的 2x3 仿射矩阵。"""
    h, w = image.shape[:2]
    center = ((w - 1) * 0.5, (h - 1) * 0.5)
    src_to_aug = cv2.getRotationMatrix2D(center, angle_deg, scale)
    aug = cv2.warpAffine(
        image,
        src_to_aug,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    aug_to_src = cv2.invertAffineTransform(src_to_aug)
    return aug, aug_to_src


def load_xfeat(xfeat_root: Path, top_k: int, force_cpu: bool, weights_path: Path | None = None):
    if not xfeat_root.exists():
        raise FileNotFoundError(
            f"XFeat 源码目录不存在：{xfeat_root}\n"
            "请先下载官方仓库到该目录，或使用当前工作流里的 third_party/accelerated_features。"
        )
    sys.path.insert(0, str(xfeat_root))
    from modules.xfeat import XFeat

    if weights_path is not None and weights_path.exists():
        ckpt = torch.load(str(weights_path), map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            weights = ckpt["model_state"]
            print(f"[load_xfeat] 使用 best/latest checkpoint（含 model_state）: {weights_path}")
        else:
            weights = str(weights_path)
            print(f"[load_xfeat] 使用自定义权重：{weights_path}")
    else:
        weights = str(xfeat_root / "weights" / "xfeat.pt")
        if not Path(weights).exists():
            raise FileNotFoundError(f"XFeat 权重不存在：{weights}")

    if force_cpu:
        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            model = XFeat(weights=weights, top_k=top_k)
        finally:
            torch.cuda.is_available = original
        return model

    return XFeat(weights=weights, top_k=top_k)


@torch.inference_mode()
def extract_features(xfeat, image: np.ndarray, top_k: int) -> dict[str, torch.Tensor]:
    # XFeat 内部会把 numpy 灰度图转成 BCHW，并 resize 到 32 的倍数。
    return xfeat.detectAndCompute(image, top_k=top_k)[0]


@torch.inference_mode()
def mutual_nn_matches(
    sar_desc: torch.Tensor,
    opt_desc: torch.Tensor,
    min_cossim: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sar_desc.numel() == 0 or opt_desc.numel() == 0:
        empty_i = np.empty((0,), dtype=np.int64)
        empty_f = np.empty((0,), dtype=np.float32)
        return empty_i, empty_i, empty_f

    sim = sar_desc @ opt_desc.t()
    best_opt_score, best_opt = sim.max(dim=1)
    best_sar = sim.max(dim=0).indices
    sar_idx = torch.arange(best_opt.shape[0], device=best_opt.device)
    keep = best_sar[best_opt] == sar_idx
    if min_cossim > -1:
        keep = keep & (best_opt_score >= min_cossim)

    sar_keep = sar_idx[keep]
    opt_keep = best_opt[keep]
    scores = best_opt_score[keep]
    return sar_keep.cpu().numpy(), opt_keep.cpu().numpy(), scores.cpu().numpy()


def estimate_transform(
    sar_pts: np.ndarray,
    opt_pts: np.ndarray,
    model: str,
    ransac_thr: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(sar_pts) < 4:
        return None, None

    if model == "similarity":
        matrix, inliers = cv2.estimateAffinePartial2D(
            sar_pts,
            opt_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thr,
            maxIters=3000,
            confidence=0.995,
            refineIters=10,
        )
        return matrix, inliers

    if model == "affine":
        matrix, inliers = cv2.estimateAffine2D(
            sar_pts,
            opt_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thr,
            maxIters=3000,
            confidence=0.995,
            refineIters=10,
        )
        return matrix, inliers

    matrix, inliers = cv2.findHomography(sar_pts, opt_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr)
    return matrix, inliers


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (2, 3):
        homo = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
        return (matrix @ homo.T).T

    pts = points.reshape(1, -1, 2).astype(np.float64)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def rmse_transform(values: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    """对 RMSE 数组做鲁棒聚合变换，降低极端大值的拉偏影响。"""
    if mode == "none":
        return values
    if mode == "cap":
        return np.clip(values, None, threshold)
    # mode == "log": 对数压缩，超出 threshold 的部分平滑压缩
    out = values.copy()
    mask = values > threshold
    if mask.any():
        out[mask] = threshold + threshold * np.log(1.0 + (values[mask] - threshold) / threshold)
    return out


def corner_rmse(matrix: np.ndarray | None, width: int, height: int, gt_aug_to_opt: np.ndarray) -> float:
    if matrix is None:
        return math.nan
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    pred = transform_points(corners, matrix)
    truth = transform_points(corners, gt_aug_to_opt)
    return float(np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=1))))


def affine_stats(matrix: np.ndarray | None) -> tuple[float, float, float, float]:
    if matrix is None or matrix.shape != (2, 3):
        return math.nan, math.nan, math.nan, math.nan
    a = float(matrix[0, 0])
    c = float(matrix[1, 0])
    scale = math.sqrt(a * a + c * c)
    angle = math.degrees(math.atan2(c, a))
    return float(matrix[0, 2]), float(matrix[1, 2]), scale, angle


def draw_match_image(
    opt_img: np.ndarray,
    sar_img: np.ndarray,
    opt_pts: np.ndarray,
    sar_pts: np.ndarray,
    inliers: np.ndarray | None,
    matrix: np.ndarray | None,
    gt_aug_to_opt: np.ndarray,
    out_path: Path,
    title: str,
    max_draw: int,
) -> None:
    opt_vis = cv2.cvtColor(opt_img, cv2.COLOR_GRAY2BGR)
    sar_vis = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)

    gap = 24
    h = max(opt_vis.shape[0], sar_vis.shape[0])
    w = opt_vis.shape[1] + gap + sar_vis.shape[1]
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    canvas[: opt_vis.shape[0], : opt_vis.shape[1]] = opt_vis
    canvas[: sar_vis.shape[0], opt_vis.shape[1] + gap :] = sar_vis

    mask = np.zeros((len(sar_pts),), dtype=bool) if inliers is None else inliers.reshape(-1).astype(bool)
    order = np.argsort(mask.astype(np.int32))
    if len(order) > max_draw:
        # 先画外点，再画内点，避免红线压住绿线。
        outlier_idx = order[~mask[order]][: max_draw // 2]
        inlier_idx = order[mask[order]][: max_draw - len(outlier_idx)]
        order = np.concatenate([outlier_idx, inlier_idx])

    x_offset = opt_vis.shape[1] + gap
    for idx in order:
        pt_opt = tuple(np.round(opt_pts[idx]).astype(int))
        pt_sar = tuple(np.round(sar_pts[idx] + np.array([x_offset, 0])).astype(int))
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
    if matrix is not None:
        pred_poly = np.round(transform_points(sar_corners, matrix)).astype(np.int32)
        cv2.polylines(canvas, [pred_poly], True, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def make_gt_aug_to_opt(offset_x: float, offset_y: float, aug_to_src: np.ndarray) -> np.ndarray:
    gt = aug_to_src.astype(np.float64).copy()
    gt[0, 2] += offset_x
    gt[1, 2] += offset_y
    return gt


def selected_labels(labels: Iterable[LabelRow], ids: list[str] | None, limit: int) -> list[LabelRow]:
    rows = list(labels)
    if ids:
        wanted = set(ids)
        rows = [row for row in rows if row.image_id in wanted]
    return rows[:limit] if limit and limit > 0 else rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = selected_labels(read_labels(args.data_root / "label.txt"), args.ids, args.limit)
    angles = parse_float_list(args.angles)
    scales = parse_float_list(args.scales)
    xfeat = load_xfeat(args.xfeat_root, args.top_k, args.force_cpu, weights_path=args.weights)

    csv_path = args.output_dir / f"xfeat_{args.preprocess}_{args.model}_summary.csv"
    fieldnames = [
        "image_id",
        "angle_deg",
        "synthetic_scale",
        "preprocess",
        "model",
        "top_k",
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

                opt_feat = extract_features(xfeat, opt, args.top_k)
                sar_feat = extract_features(xfeat, sar, args.top_k)
                sar_idx, opt_idx, _scores = mutual_nn_matches(
                    sar_feat["descriptors"],
                    opt_feat["descriptors"],
                    args.min_cossim,
                )

                sar_pts = sar_feat["keypoints"][sar_idx].cpu().numpy().astype(np.float32)
                opt_pts = opt_feat["keypoints"][opt_idx].cpu().numpy().astype(np.float32)
                matrix, inliers = estimate_transform(sar_pts, opt_pts, args.model, args.ransac_thr)

                inlier_count = int(inliers.sum()) if inliers is not None else 0
                match_count = int(len(sar_pts))
                inlier_ratio = (inlier_count / match_count) if match_count else 0.0
                accepted = inlier_count >= args.min_accept_inliers and inlier_ratio >= args.min_accept_ratio
                rmse = corner_rmse(matrix, sar.shape[1], sar.shape[0], gt_aug_to_opt)
                tx, ty, est_scale, est_angle = affine_stats(matrix)

                stem = f"{label.image_id}_a{angle:g}_s{scale:g}_{args.preprocess}_{args.model}.jpg"
                draw_path = args.output_dir / "matches" / stem
                if not args.no_images:
                    title = (
                        f"id={label.image_id} matches={match_count} inliers={inlier_count} "
                        f"rmse={rmse:.2f}px angle={angle:g} scale={scale:g}"
                    )
                    draw_match_image(
                        opt,
                        sar,
                        opt_pts,
                        sar_pts,
                        inliers,
                        matrix,
                        gt_aug_to_opt,
                        draw_path,
                        title,
                        args.max_draw,
                    )

                row = {
                    "image_id": label.image_id,
                    "angle_deg": angle,
                    "synthetic_scale": scale,
                    "preprocess": args.preprocess,
                    "model": args.model,
                    "top_k": args.top_k,
                    "matches": match_count,
                    "inliers": inlier_count,
                    "inlier_ratio": inlier_ratio,
                    "accepted": int(accepted),
                    "corner_rmse": rmse,
                    "tx": tx,
                    "ty": ty,
                    "estimated_scale": est_scale,
                    "estimated_angle_deg": est_angle,
                    "draw_path": str(draw_path) if not args.no_images else "",
                }
                rows.append(row)
                print(
                    f"{label.image_id} angle={angle:g} scale={scale:g}: "
                    f"matches={match_count}, inliers={inlier_count}, rmse={rmse:.2f}px"
                )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rmse = np.array([float(row["corner_rmse"]) for row in rows if not math.isnan(float(row["corner_rmse"]))])
    print(f"\nCSV: {csv_path}")
    if len(valid_rmse):
        # Transform RMSE for robust mean computation
        mode = args.rmse_transform
        thr = args.rmse_threshold
        transformed = rmse_transform(valid_rmse, mode, thr)
        robust_mean = np.mean(transformed)

        if mode == "none":
            label = f"mean"
        elif mode == "cap":
            label = f"mean_cap@{thr:g}"
        else:
            label = f"mean_log@{thr:g}"

        p95 = float(np.percentile(valid_rmse, 95))
        print(
            "Summary: "
            f"cases={len(rows)}, valid={len(valid_rmse)}, "
            f"<=5px={(valid_rmse <= 5).sum()}, <=10px={(valid_rmse <= 10).sum()}, "
            f"median={np.median(valid_rmse):.2f}px, "
            f"mean={np.mean(valid_rmse):.2f}px"
        )
        if mode != "none":
            n_affected = int((valid_rmse > thr).sum())
            print(
                f"         "
                f"{label}={robust_mean:.2f}px  "
                f"P95={p95:.2f}px  "
                f"affected={n_affected}/{len(valid_rmse)} samples"
            )
        accepted_rmse = np.array(
            [
                float(row["corner_rmse"])
                for row in rows
                if int(row["accepted"]) == 1 and not math.isnan(float(row["corner_rmse"]))
            ]
        )
        if args.min_accept_inliers > 0 or args.min_accept_ratio > 0:
            if len(accepted_rmse):
                p95_acc = float(np.percentile(accepted_rmse, 95))
                print(
                    "Accepted: "
                    f"cases={len(accepted_rmse)}, "
                    f"<=5px={(accepted_rmse <= 5).sum()}, <=10px={(accepted_rmse <= 10).sum()}, "
                    f"median={np.median(accepted_rmse):.2f}px, mean={np.mean(accepted_rmse):.2f}px"
                )
                print(f"         P95={p95_acc:.2f}px")
            else:
                print("Accepted: cases=0")
    else:
        print(f"Summary: cases={len(rows)}, valid=0")


if __name__ == "__main__":
    main()
