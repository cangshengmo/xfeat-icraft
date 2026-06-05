"""
Export the XFeat frontend network to fixed-size ONNX models.

Only XFeatModel.forward is exported:
    image -> dense_descriptors, keypoint_logits, reliability

Dynamic post-processing such as heatmap decoding, NMS, top-k selection,
descriptor interpolation, mutual-NN matching and RANSAC stays on the host.
This keeps the ICRAFT compilation target close to a plain CNN frontend.

Use the torch201 environment for this script.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "xfeat_onnx"
DEFAULT_ICRAFT_DIR = REPO_DIR / "icraft_compile"

PRESETS = {
    "sar512": {"height": 512, "width": 512, "role": "sar"},
    "opt800": {"height": 800, "width": 800, "role": "opt"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-size XFeat frontend ONNX models.")
    parser.add_argument(
        "--preset",
        choices=["sar512", "opt800", "both"],
        default="both",
        help="Named export target. Use both to export 512x512 and 800x800.",
    )
    parser.add_argument("--height", type=int, default=None, help="Custom fixed input height.")
    parser.add_argument("--width", type=int, default=None, help="Custom fixed input width.")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="Input channels. Deployment uses grayscale.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset. ICRAFT v3.6.2 docs mention PyTorch opset17.")
    parser.add_argument("--weights", type=Path, default=REPO_DIR / "weights" / "xfeat.pt", help="XFeat weight file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for ONNX files.")
    parser.add_argument("--output", type=Path, default=None, help="Single custom output path. Only valid with one target.")
    parser.add_argument("--check", action="store_true", help="Run ONNXRuntime numerical alignment check after export.")
    parser.add_argument(
        "--sync-icraft",
        action="store_true",
        help="Copy exported ONNX files into icraft_compile/xfeat_512/model and xfeat_800/model.",
    )
    parser.add_argument("--icraft-dir", type=Path, default=DEFAULT_ICRAFT_DIR, help="ICRAFT compile root.")
    return parser.parse_args()


def selected_targets(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.height is not None or args.width is not None:
        if args.preset == "both":
            raise ValueError("Custom --height/--width cannot be combined with --preset both.")
        if args.height is None or args.width is None:
            raise ValueError("Custom export needs both --height and --width.")
        return [{"name": f"custom_{args.height}x{args.width}", "height": args.height, "width": args.width, "role": "custom"}]

    names = ["sar512", "opt800"] if args.preset == "both" else [args.preset]
    return [{"name": name, **PRESETS[name]} for name in names]


def load_model(weights: Path) -> torch.nn.Module:
    if not weights.exists():
        raise FileNotFoundError(f"XFeat weights not found: {weights}")
    sys.path.insert(0, str(REPO_DIR))

    from modules.model import XFeatModel

    model = XFeatModel().eval()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    return model


def check_export(model: torch.nn.Module, onnx_path: Path, dummy: torch.Tensor) -> None:
    with torch.inference_mode():
        torch_outputs = [out.detach().cpu().numpy() for out in model(dummy)]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"image": dummy.cpu().numpy().astype(np.float32)})

    names = ["dense_descriptors", "keypoint_logits", "reliability"]
    for name, torch_out, ort_out in zip(names, torch_outputs, ort_outputs):
        diff = np.abs(torch_out - ort_out)
        rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(torch_out), 1e-12)
        print(
            f"  {name}: shape={ort_out.shape}, "
            f"max_abs={diff.max():.6e}, mean_abs={diff.mean():.6e}, rel_l2={rel_l2:.6e}"
        )


def output_path_for(args: argparse.Namespace, target: dict[str, object], target_count: int) -> Path:
    if args.output is not None:
        if target_count != 1:
            raise ValueError("--output can only be used when exporting one target.")
        return args.output
    height = int(target["height"])
    width = int(target["width"])
    return args.output_dir / f"xfeat_frontend_{height}x{width}.onnx"


def sync_to_icraft(onnx_path: Path, target: dict[str, object], icraft_dir: Path) -> None:
    height = int(target["height"])
    workspace = icraft_dir / f"xfeat_{height}"
    dst = workspace / "model" / onnx_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dst)
    print(f"  synced to ICRAFT workspace: {dst}")


def export_one(model: torch.nn.Module, args: argparse.Namespace, target: dict[str, object], target_count: int) -> Path:
    height = int(target["height"])
    width = int(target["width"])
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("XFeat input size should be divisible by 32 for stable pyramid fusion.")

    output = output_path_for(args, target, target_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, args.channels, height, width, dtype=torch.float32)

    print(f"Exporting {height}x{width} -> {output}")
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["image"],
        output_names=["dense_descriptors", "keypoint_logits", "reliability"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker passed")
    print(f"  dense_descriptors: [1, 64, {height // 8}, {width // 8}]")
    print(f"  keypoint_logits:   [1, 65, {height // 8}, {width // 8}]")
    print(f"  reliability:       [1, 1,  {height // 8}, {width // 8}]")

    if args.check:
        check_export(model, output, dummy)
    if args.sync_icraft:
        sync_to_icraft(output, target, args.icraft_dir)
    return output


def main() -> None:
    args = parse_args()
    targets = selected_targets(args)
    model = load_model(args.weights)
    outputs = [export_one(model, args, target, len(targets)) for target in targets]
    print("")
    print("Exported ONNX files:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
