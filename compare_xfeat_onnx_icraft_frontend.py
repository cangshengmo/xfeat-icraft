"""
Compare one compiled XFeat ICRAFT frontend with its ONNX model.

This script compares only frontend outputs:
  dense_descriptors, keypoint_logits, reliability

Host-side XFeat post-processing and SAR/OPT matching are intentionally not
included here.

Use the torch201 environment with the ICRAFT SDK available.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_COMPILE_ROOT = REPO_DIR / "icraft_compile"
DEFAULT_URL = "socket://ql100aiu@192.168.137.50:9981?npu=0x40000000&dma=0x80000000"

TARGETS = {
    "sar512": {"workspace": "xfeat_512", "net": "xfeat_frontend_512", "height": 512, "width": 512},
    "opt800": {"workspace": "xfeat_800", "net": "xfeat_frontend_800", "height": 800, "width": 800},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XFeat ONNX and ICRAFT frontend outputs.")
    parser.add_argument("--compile-root", type=Path, default=DEFAULT_COMPILE_ROOT)
    parser.add_argument("--target", choices=["sar512", "opt800"], default="sar512")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--image-id", default="", help="Defaults to the first entry in qtset/ftmp.txt.")
    parser.add_argument("--input-type", choices=["ftmp", "bmp"], default="ftmp")
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--scale", choices=["auto", "raw255", "unit"], default="auto")
    parser.add_argument("--save-csv", type=Path, default=REPO_DIR / "outputs" / "xfeat_onnx_vs_icraft_frontend.csv")
    return parser.parse_args()


def workspace_for(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    spec = TARGETS[args.target]
    workspace = args.workspace or (args.compile_root / str(spec["workspace"]))
    return workspace, spec


def first_id(workspace: Path) -> str:
    list_path = workspace / "qtset" / "ftmp.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing quantization list: {list_path}")
    for line in list_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item:
            return Path(item).stem
    raise ValueError(f"No entries in {list_path}")


def deploy_scale(workspace: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    config_path = workspace / "deploy" / "deploy_config.json"
    if not config_path.exists():
        return "raw255"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("input_value_scale", "raw255")


def default_paths(args: argparse.Namespace, workspace: Path, spec: dict[str, object]) -> tuple[Path, Path, Path]:
    net = str(spec["net"])
    onnx = args.onnx or next((workspace / "model").glob("*.onnx"))
    json_path = args.json or (workspace / "imodels" / net / f"{net}_BY.json")
    raw_path = args.raw or (workspace / "imodels" / net / f"{net}_BY.raw")
    return onnx, json_path, raw_path


def read_input(workspace: Path, image_id: str, input_type: str, height: int, width: int, scale: str) -> tuple[np.ndarray, np.ndarray]:
    if input_type == "ftmp":
        path = workspace / "qtset" / "ftmp" / f"{image_id}.ftmp"
        data = np.fromfile(str(path), dtype=np.float32)
        expected = height * width
        if data.size != expected:
            raise ValueError(f"Unexpected FTMP size for {path}: got {data.size}, expected {expected}")
        image = data.reshape(height, width).astype(np.float32)
        preview = np.clip(image if scale == "raw255" else image * 255.0, 0, 255).astype(np.uint8)
    else:
        path = workspace / "qtset" / "bmp" / f"{image_id}.bmp"
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if gray.shape[:2] != (height, width):
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        image = gray.astype(np.float32)
        if scale == "unit":
            image = image / 255.0
        preview = gray

    nhwc = image.reshape(1, height, width, 1).astype(np.float32)
    nchw = np.transpose(nhwc, (0, 3, 1, 2)).astype(np.float32)
    return nhwc, nchw


def normalize_output(name: str, array: np.ndarray, height: int, width: int) -> np.ndarray:
    channels = {"dense_descriptors": 64, "keypoint_logits": 65, "reliability": 1}[name]
    out_h, out_w = height // 8, width // 8
    nchw = (1, channels, out_h, out_w)
    nhwc = (1, out_h, out_w, channels)
    if array.shape == nchw:
        return array.astype(np.float32, copy=False)
    if array.shape == nhwc:
        return np.transpose(array, (0, 3, 1, 2)).astype(np.float32, copy=False)
    raise ValueError(f"{name} shape {array.shape} does not match {nchw} or {nhwc}")


def tensor_diff(name: str, onnx_out: np.ndarray, icraft_out: np.ndarray) -> dict[str, object]:
    diff = np.abs(onnx_out.astype(np.float64) - icraft_out.astype(np.float64))
    base = max(np.linalg.norm(onnx_out.astype(np.float64)), 1e-12)
    return {
        "name": name,
        "shape": str(tuple(onnx_out.shape)),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l2": float(np.linalg.norm(diff) / base),
    }


def run_onnx(onnx_path: Path, nchw: np.ndarray) -> tuple[list[np.ndarray], float]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    outputs = session.run(None, {session.get_inputs()[0].name: nchw})
    return outputs, time.perf_counter() - start


def run_icraft(json_path: Path, raw_path: Path, nhwc: np.ndarray, url: str) -> tuple[list[np.ndarray], float]:
    from icraft.xir import Network
    from icraft.xrt import Device, Layout, Session, Tensor
    from icraft.host_backend import HostBackend, HostDevice
    from icraft.buyibackend import BuyiBackend

    device = Device.Open(url)
    try:
        network = Network.CreateFromJsonFile(str(json_path))
        network.loadParamsFromFile(str(raw_path))
        session = Session.Create([BuyiBackend, HostBackend], network.view(0), [device, HostDevice.Default()])
        session.apply()
        start = time.perf_counter()
        outputs = session.forward([Tensor(nhwc.astype(np.float32), Layout("NHWC"))])
        device.reset(1)
        elapsed = time.perf_counter() - start
        return [np.array(out) for out in outputs], elapsed
    finally:
        Device.Close(device)


def save_csv(path: Path, image_id: str, target: str, onnx_s: float, icraft_s: float, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "target", "output", "shape", "max_abs", "mean_abs", "rel_l2", "onnx_s", "icraft_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": image_id,
                    "target": target,
                    "output": row["name"],
                    "shape": row["shape"],
                    "max_abs": "{:.6e}".format(row["max_abs"]),
                    "mean_abs": "{:.6e}".format(row["mean_abs"]),
                    "rel_l2": "{:.6e}".format(row["rel_l2"]),
                    "onnx_s": "{:.6f}".format(onnx_s),
                    "icraft_s": "{:.6f}".format(icraft_s),
                }
            )


def main() -> None:
    args = parse_args()
    workspace, spec = workspace_for(args)
    image_id = args.image_id or first_id(workspace)
    scale = deploy_scale(workspace, args.scale)
    onnx_path, json_path, raw_path = default_paths(args, workspace, spec)
    height, width = int(spec["height"]), int(spec["width"])

    nhwc, nchw = read_input(workspace, image_id, args.input_type, height, width, scale)
    onnx_outputs, onnx_s = run_onnx(onnx_path, nchw)
    icraft_outputs, icraft_s = run_icraft(json_path, raw_path, nhwc, args.url)

    names = ["dense_descriptors", "keypoint_logits", "reliability"]
    rows: list[dict[str, object]] = []
    for name, onnx_out, icraft_out in zip(names, onnx_outputs, icraft_outputs):
        onnx_norm = normalize_output(name, onnx_out, height, width)
        icraft_norm = normalize_output(name, icraft_out, height, width)
        row = tensor_diff(name, onnx_norm, icraft_norm)
        rows.append(row)
        print(
            f"{name}: shape={row['shape']} max_abs={row['max_abs']:.6e} "
            f"mean_abs={row['mean_abs']:.6e} rel_l2={row['rel_l2']:.6e}"
        )

    print(f"timing: onnx={onnx_s:.4f}s icraft={icraft_s:.4f}s")
    save_csv(args.save_csv, image_id, args.target, onnx_s, icraft_s, rows)
    print(f"saved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
