"""
Run ICRAFT compile stages for XFeat frontend workspaces.

Use the torch201 environment for this script:
  conda run -n torch201 python compile_xfeat_icraft.py --target both
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_COMPILE_ROOT = REPO_DIR / "icraft_compile"
DEFAULT_STAGES = ["parse", "optimize", "quantize", "adapt", "generate"]

TARGETS = {
    "sar512": ("xfeat_512", "xfeat_512.toml", "xfeat_frontend_512"),
    "opt800": ("xfeat_800", "xfeat_800.toml", "xfeat_frontend_800"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile XFeat frontend ONNX models with ICRAFT.")
    parser.add_argument("--compile-root", type=Path, default=DEFAULT_COMPILE_ROOT)
    parser.add_argument("--target", choices=["sar512", "opt800", "both"], default="both")
    parser.add_argument("--icraft", default="icraft.exe", help="Path to icraft.exe.")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, choices=["parse", "optimize", "quantize", "adapt", "generate", "compile", "run"])
    parser.add_argument("--clean", action="store_true", help="Remove .icraft and imodels inside selected workspaces before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--require-torch201", action="store_true", help="Fail if the current Python prefix is not torch201.")
    return parser.parse_args()


def selected_targets(name: str) -> list[tuple[str, str, str]]:
    if name == "both":
        return [TARGETS["sar512"], TARGETS["opt800"]]
    return [TARGETS[name]]


def check_env(require: bool) -> None:
    env_name = Path(sys.prefix).name.lower()
    if env_name != "torch201":
        message = f"[warn] Current Python env is '{Path(sys.prefix).name}', expected 'torch201' for ONNX/ICRAFT work."
        if require:
            raise RuntimeError(message)
        print(message)


def ensure_inside(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing to clean outside compile root: {resolved_path}")


def clean_workspace(workspace: Path, compile_root: Path) -> None:
    for name in [".icraft", "imodels"]:
        target = workspace / name
        if not target.exists():
            continue
        ensure_inside(target, compile_root)
        shutil.rmtree(target)
        print(f"  removed {target}")


def validate_workspace(workspace: Path, toml_name: str) -> None:
    toml = workspace / toml_name
    if not toml.exists():
        raise FileNotFoundError(f"TOML not found: {toml}")
    model_dir = workspace / "model"
    if not model_dir.is_dir() or not list(model_dir.glob("*.onnx")):
        raise FileNotFoundError(f"No ONNX found in {model_dir}")
    ftmp_list = workspace / "qtset" / "ftmp.txt"
    if not ftmp_list.exists():
        raise FileNotFoundError(f"Quantization list not found: {ftmp_list}")


def run_stage(icraft: str, stage: str, workspace: Path, toml_name: str, dry_run: bool) -> None:
    cmd = [icraft, stage, toml_name]
    print(f"[{workspace.name}] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(workspace), check=True)


def final_outputs(workspace: Path, net_name: str) -> tuple[Path, Path]:
    base = workspace / "imodels" / net_name
    return base / f"{net_name}_BY.json", base / f"{net_name}_BY.raw"


def compile_target(target: tuple[str, str, str], args: argparse.Namespace) -> None:
    workspace_name, toml_name, net_name = target
    workspace = args.compile_root / workspace_name
    validate_workspace(workspace, toml_name)
    if args.clean:
        clean_workspace(workspace, args.compile_root)

    for stage in args.stages:
        run_stage(args.icraft, stage, workspace, toml_name, args.dry_run)

    json_path, raw_path = final_outputs(workspace, net_name)
    if not args.dry_run and "generate" in args.stages:
        print(f"[{workspace.name}] expected outputs:")
        print(f"  {json_path} {'OK' if json_path.exists() else 'MISSING'}")
        print(f"  {raw_path} {'OK' if raw_path.exists() else 'MISSING'}")


def main() -> None:
    args = parse_args()
    check_env(args.require_torch201)
    for target in selected_targets(args.target):
        compile_target(target, args)


if __name__ == "__main__":
    main()
