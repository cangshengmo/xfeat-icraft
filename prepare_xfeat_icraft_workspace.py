"""
Prepare XFeat ICRAFT compile workspaces.

The XFeat deployment boundary is a single-image frontend. This script creates
two compile workspaces by default:

  icraft_compile/xfeat_512  -> SAR-sized 512x512 frontend
  icraft_compile/xfeat_800  -> OPT-sized 800x800 frontend

Each workspace contains a model directory, quantization BMP/FTMP samples,
list files and a TOML compile config.

Use the torch201 environment for this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\HanZhQ\PCIE715\Project_Trans\2350")
DEFAULT_ONNX_DIR = REPO_DIR / "outputs" / "xfeat_onnx"
DEFAULT_COMPILE_ROOT = REPO_DIR / "icraft_compile"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    role: str
    source_subdir: str
    height: int
    width: int

    @property
    def workspace_name(self) -> str:
        return f"xfeat_{self.height}"

    @property
    def net_name(self) -> str:
        return f"xfeat_frontend_{self.height}"

    @property
    def onnx_name(self) -> str:
        return f"xfeat_frontend_{self.height}x{self.width}.onnx"

    @property
    def toml_name(self) -> str:
        return f"xfeat_{self.height}.toml"


TARGETS = {
    "sar512": TargetSpec("sar512", "sar", "sar", 512, 512),
    "opt800": TargetSpec("opt800", "opt", "opt", 800, 800),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare XFeat ICRAFT compile workspaces.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--compile-root", type=Path, default=DEFAULT_COMPILE_ROOT)
    parser.add_argument("--target", choices=["sar512", "opt800", "both"], default="both")
    parser.add_argument("--limit", type=int, default=10, help="Number of label ids to use for quantization samples.")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional explicit image ids.")
    parser.add_argument("--preprocess", choices=["raw", "clahe", "grad", "canny"], default="grad")
    parser.add_argument("--sar-blur", type=int, default=3)
    parser.add_argument(
        "--ftmp-scale",
        choices=["raw255", "unit"],
        default="raw255",
        help="raw255 matches current XFeat PyTorch inference; unit writes image/255.0.",
    )
    parser.add_argument("--no-copy-onnx", action="store_true", help="Do not copy ONNX files into workspaces.")
    return parser.parse_args()


def selected_targets(name: str) -> list[TargetSpec]:
    if name == "both":
        return [TARGETS["sar512"], TARGETS["opt800"]]
    return [TARGETS[name]]


def read_label_ids(data_root: Path, ids: list[str] | None, limit: int) -> list[str]:
    if ids:
        return ids[:limit] if limit > 0 else ids
    label_path = data_root / "label.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"label.txt not found: {label_path}")
    out: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            out.append(parts[0])
        if limit > 0 and len(out) >= limit:
            break
    return out


def find_image(root: Path, subdir: str, image_id: str) -> Path:
    folder = root / subdir
    for ext in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = folder / f"{image_id}{ext}"
        if path.exists():
            return path
    matches = sorted(folder.glob(f"{image_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image not found: {folder / image_id}.*")


def preprocess_gray(image: np.ndarray, mode: str, is_sar: bool, sar_blur: int) -> np.ndarray:
    out = image
    if is_sar and sar_blur and sar_blur > 1:
        kernel = sar_blur if sar_blur % 2 == 1 else sar_blur + 1
        out = cv2.medianBlur(out, kernel)

    if mode == "raw":
        return out
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
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
    raise ValueError(f"Unknown preprocess mode: {mode}")


def read_processed_image(data_root: Path, spec: TargetSpec, image_id: str, preprocess: str, sar_blur: int) -> np.ndarray:
    path = find_image(data_root, spec.source_subdir, image_id)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = preprocess_gray(image, preprocess, is_sar=(spec.role == "sar"), sar_blur=sar_blur)
    if image.shape[:2] != (spec.height, spec.width):
        image = cv2.resize(image, (spec.width, spec.height), interpolation=cv2.INTER_LINEAR)
    return image


def write_ftmp(image: np.ndarray, output: Path, scale: str) -> None:
    values = image.astype(np.float32)
    if scale == "unit":
        values = values / 255.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(values.reshape(1, image.shape[0], image.shape[1]).tobytes())


def write_toml(workspace: Path, spec: TargetSpec, first_id: str) -> Path:
    toml = f"""[parse]
pre_check = false
net_name = "{spec.net_name}"
framework = "onnx"
inputs = [[1, {spec.height}, {spec.width}, 1]]
inputs_layout = "NHWC"
pre_method = 'nop'
pre_scale = 'nop'
pre_mean = 'nop'
channel_swap = 'nop'
network = "./model/{spec.onnx_name}"
jr_path = "./imodels/{spec.net_name}/"

[optimize]
target = "BUYI"
json = "./imodels/{spec.net_name}/{spec.net_name}_parsed.json"
raw = "./imodels/{spec.net_name}/{spec.net_name}_parsed.raw"
jr_path = "./imodels/{spec.net_name}/"

[quantize]
forward_mode = "image"
saturation = "kld"
forward_dir = "./qtset/ftmp/"
forward_list = "./qtset/ftmp.txt"
bits = 8
json = "./imodels/{spec.net_name}/{spec.net_name}_optimized.json"
raw = "./imodels/{spec.net_name}/{spec.net_name}_optimized.raw"
jr_path = "./imodels/{spec.net_name}/"
per = "tensor"
target = "buyi"
no_transinput = true
no_imagemake = true

[adapt]
target = "BUYI"
json = "./imodels/{spec.net_name}/{spec.net_name}_quantized.json"
raw = "./imodels/{spec.net_name}/{spec.net_name}_quantized.raw"
jr_path = "./imodels/{spec.net_name}/"

[generate]
json = "./imodels/{spec.net_name}/{spec.net_name}_adapted.json"
raw = "./imodels/{spec.net_name}/{spec.net_name}_adapted.raw"
jr_path = "./imodels/{spec.net_name}/"
etmopt = 0
log_path = "./logs/"
no_mergeops = true

[run]
log_time = true
log_io = true
json = "./imodels/{spec.net_name}/{spec.net_name}_BY.json"
raw = "./imodels/{spec.net_name}/{spec.net_name}_BY.raw"
input = "./qtset/bmp/{first_id}.bmp"
backends = "Host"
"""
    path = workspace / spec.toml_name
    path.write_text(toml, encoding="utf-8")
    return path


def copy_onnx_if_available(onnx_dir: Path, workspace: Path, spec: TargetSpec) -> None:
    src = onnx_dir / spec.onnx_name
    dst = workspace / "model" / spec.onnx_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"[warn] ONNX not found, skip copy: {src}")
        return
    shutil.copy2(src, dst)
    print(f"  copied ONNX: {dst}")


def write_deploy_config(workspace: Path, spec: TargetSpec, args: argparse.Namespace, ids: list[str]) -> None:
    config = {
        "model_type": "xfeat_frontend",
        "role": spec.role,
        "input_format": "grayscale_float32_nchw_export_nhwc_icraft",
        "input_value_scale": args.ftmp_scale,
        "preprocess": args.preprocess,
        "sar_blur": args.sar_blur if spec.role == "sar" else 0,
        "input_hw": [spec.height, spec.width],
        "feature_stride": 8,
        "outputs": {
            "dense_descriptors": [1, 64, spec.height // 8, spec.width // 8],
            "keypoint_logits": [1, 65, spec.height // 8, spec.width // 8],
            "reliability": [1, 1, spec.height // 8, spec.width // 8],
        },
        "host_postprocess": ["heatmap_decode", "nms", "top_k", "descriptor_interpolation", "mutual_nn", "similarity_ransac"],
        "quantization_ids": ids,
    }
    deploy_dir = workspace / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "deploy_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def prepare_target(spec: TargetSpec, ids: list[str], args: argparse.Namespace) -> None:
    workspace = args.compile_root / spec.workspace_name
    bmp_dir = workspace / "qtset" / "bmp"
    ftmp_dir = workspace / "qtset" / "ftmp"
    bmp_dir.mkdir(parents=True, exist_ok=True)
    ftmp_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "model").mkdir(parents=True, exist_ok=True)

    bmp_lines: list[str] = []
    ftmp_lines: list[str] = []
    for image_id in ids:
        image = read_processed_image(args.data_root, spec, image_id, args.preprocess, args.sar_blur)
        bmp_path = bmp_dir / f"{image_id}.bmp"
        ftmp_path = ftmp_dir / f"{image_id}.ftmp"
        cv2.imwrite(str(bmp_path), image)
        write_ftmp(image, ftmp_path, args.ftmp_scale)
        bmp_lines.append(f"{image_id}.bmp")
        ftmp_lines.append(f"{image_id}.ftmp")

    (workspace / "qtset" / "bmp.txt").write_text("\n".join(bmp_lines) + "\n", encoding="utf-8")
    (workspace / "qtset" / "ftmp.txt").write_text("\n".join(ftmp_lines) + "\n", encoding="utf-8")
    toml_path = write_toml(workspace, spec, ids[0])
    write_deploy_config(workspace, spec, args, ids)
    if not args.no_copy_onnx:
        copy_onnx_if_available(args.onnx_dir, workspace, spec)

    print(f"Prepared {spec.name}: {workspace}")
    print(f"  TOML: {toml_path}")
    print(f"  samples: {len(ids)}")


def main() -> None:
    args = parse_args()
    ids = read_label_ids(args.data_root, args.ids, args.limit)
    if not ids:
        raise ValueError("No image ids selected.")
    args.compile_root.mkdir(parents=True, exist_ok=True)

    for spec in selected_targets(args.target):
        prepare_target(spec, ids, args)


if __name__ == "__main__":
    main()
