#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Gemma 3 1B HF weights into local MaxText checkpoints.")
    parser.add_argument("mode", nargs="?", choices=("free", "scan", "both"), default="both")
    parser.add_argument("--python-bin", default=None, help="Python interpreter to use. Defaults to the current interpreter.")
    parser.add_argument("--config-path", default=None, help="MaxText config path. Defaults to src/maxtext/configs/base.yml.")
    parser.add_argument("--model-name", default="gemma3-1b")
    parser.add_argument("--hf-model-path", default="google/gemma-3-1b-it")
    parser.add_argument("--hf-access-token", default=os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN") or "")
    parser.add_argument("--save-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--eager-load-method",
        choices=("transformers", "safetensors"),
        default=None,
        help="Optional eager HF load backend. Omit to use MaxText default.",
    )
    parser.add_argument(
        "--lazy-load-tensors",
        action="store_true",
        help="Use MaxText lazy tensor loading during conversion.",
    )
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/mnt/carles/.cache"))
    parser.add_argument("--models-dir", default=None, help="Defaults to ./models under the repo root.")
    parser.add_argument("--free-out", default=None, help="Defaults to ./models/gemma31b")
    parser.add_argument("--scan-out", default=None, help="Defaults to ./models/gemma31b-scan")
    return parser


def run_convert(
    *,
    python_bin: str,
    config_path: str,
    model_name: str,
    out_dir: str,
    run_name: str,
    hf_access_token: str,
    hf_model_path: str,
    save_dtype: str,
    scan_layers: bool,
    eager_load_method: str | None,
    lazy_load_tensors: bool,
    hf_home: str,
    repo_root: Path,
) -> None:
    env = os.environ.copy()
    env.setdefault("HF_HOME", hf_home)

    args = [
        python_bin,
        "-m",
        "maxtext.checkpoint_conversion.to_maxtext",
        config_path,
        f"model_name={model_name}",
        f"base_output_directory={out_dir}",
        f"run_name={run_name}",
        f"hf_access_token={hf_access_token}",
        "hardware=cpu",
        "skip_jax_distributed_system=True",
        f"scan_layers={'true' if scan_layers else 'false'}",
        f"--save_dtype={save_dtype}",
    ]

    if hf_model_path:
        args.append(f"--hf_model_path={hf_model_path}")
    if eager_load_method:
        args.append(f"--eager_load_method={eager_load_method}")
    if lazy_load_tensors:
        args.append("--lazy_load_tensors=True")

    print(f"Converting {model_name} -> {out_dir} (scan_layers={scan_layers})")
    subprocess.run(args, check=True, cwd=repo_root, env=env)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir) if args.models_dir else repo_root / "models"
    free_out = Path(args.free_out) if args.free_out else models_dir / "gemma31b"
    scan_out = Path(args.scan_out) if args.scan_out else models_dir / "gemma31b-scan"
    python_bin = args.python_bin or sys.executable
    config_path = args.config_path or str(repo_root / "src" / "maxtext" / "configs" / "base.yml")

    models_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("free", "both"):
        run_convert(
            python_bin=python_bin,
            config_path=config_path,
            model_name=args.model_name,
            out_dir=str(free_out),
            run_name=f"{args.model_name}-to-maxtext-free",
            hf_access_token=args.hf_access_token,
            hf_model_path=args.hf_model_path,
            save_dtype=args.save_dtype,
            scan_layers=False,
            eager_load_method=args.eager_load_method,
            lazy_load_tensors=args.lazy_load_tensors,
            hf_home=args.hf_home,
            repo_root=repo_root,
        )

    if args.mode in ("scan", "both"):
        run_convert(
            python_bin=python_bin,
            config_path=config_path,
            model_name=args.model_name,
            out_dir=str(scan_out),
            run_name=f"{args.model_name}-to-maxtext-scan",
            hf_access_token=args.hf_access_token,
            hf_model_path=args.hf_model_path,
            save_dtype=args.save_dtype,
            scan_layers=True,
            eager_load_method=args.eager_load_method,
            lazy_load_tensors=args.lazy_load_tensors,
            hf_home=args.hf_home,
            repo_root=repo_root,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
