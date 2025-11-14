#!/usr/bin/env python3
"""Utility to render MANO predictions saved as ``.npz`` files back onto RGB frames."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

from body_model import MANO
from util.tensor import move_to, to_torch
from vis.output import animate_scene, prep_result_vis
from vis.viewer import init_viewer

ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _sorted_images(img_root: Path, max_frames: int) -> Sequence[str]:
    """Return a sorted list of frame paths limited to ``max_frames`` entries."""
    if not img_root.is_dir():
        raise FileNotFoundError(f"Image directory {img_root} does not exist")

    img_paths = sorted(
        str(p)
        for p in img_root.iterdir()
        if p.suffix.lower() in ALLOWED_IMG_EXTS and p.is_file()
    )
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No image files with extensions {ALLOWED_IMG_EXTS} in {img_root}")
    if len(img_paths) < max_frames:
        raise ValueError(
            f"Need at least {max_frames} images to render {max_frames} frames, found {len(img_paths)}"
        )
    return img_paths[:max_frames]


def _reshape_pose(pose: np.ndarray) -> np.ndarray:
    """Flatten MANO pose tensors to (B, T, 45) if necessary."""
    if pose.ndim == 4:
        B, T, J, C = pose.shape
        return pose.reshape(B, T, J * C)
    return pose


def _extract_camera_field(data: np.ndarray, field_name: str) -> np.ndarray:
    if data.ndim == 4:  # (B, T, 3, 3)
        return data[0]
    if data.ndim == 3 and data.shape[-1] == 3:
        return data
    raise ValueError(f"Unexpected shape {data.shape} for camera field '{field_name}'")


def _get_intrinsics(npz_data: Dict[str, np.ndarray], override: Sequence[float] | None) -> np.ndarray:
    if override is not None:
        arr = np.asarray(override, dtype=np.float32)
        if arr.shape != (4,):
            raise ValueError("--intrins must contain exactly four values: fx fy cx cy")
        return arr
    key = None
    if "intrins" in npz_data:
        key = "intrins"
    elif "intrinsics" in npz_data:
        key = "intrinsics"
    if key is not None:
        intrins = np.asarray(npz_data[key], dtype=np.float32)
        if intrins.ndim > 1:
            intrins = intrins[0]
        return intrins[:4]
    raise ValueError("Could not find camera intrinsics. Pass --intrins to provide them explicitly.")


def _build_result_dict(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    npz_data = {k: data[k] for k in data.files}

    required = ["pose_body", "root_orient", "trans", "is_right"]
    missing = [k for k in required if k not in npz_data]
    if missing:
        raise KeyError(f"Missing required fields in {npz_path}: {missing}")

    pose_body = _reshape_pose(npz_data["pose_body"])
    trans = np.asarray(npz_data["trans"], dtype=np.float32)
    root_orient = np.asarray(npz_data["root_orient"], dtype=np.float32)
    is_right = np.asarray(npz_data["is_right"], dtype=np.float32)

    betas = np.asarray(npz_data.get("betas")) if "betas" in npz_data else None

    result = {
        "pose_body": pose_body.astype(np.float32),
        "trans": trans,
        "root_orient": root_orient,
        "is_right": is_right,
    }
    if betas is not None:
        result["betas"] = betas.astype(np.float32)
    if "intrins" in npz_data:
        result["intrins"] = np.asarray(npz_data["intrins"], dtype=np.float32)
    if "intrinsics" in npz_data:
        result["intrinsics"] = np.asarray(npz_data["intrinsics"], dtype=np.float32)

    if "cam_R" in npz_data or "camR" in npz_data:
        key = "cam_R" if "cam_R" in npz_data else "camR"
        result["cam_R"] = _extract_camera_field(npz_data[key], key).astype(np.float32)
    if "cam_t" in npz_data or "camt" in npz_data:
        key = "cam_t" if "cam_t" in npz_data else "camt"
        cam_t = np.asarray(npz_data[key], dtype=np.float32)
        if cam_t.ndim == 3:
            cam_t = cam_t[0]
        result["cam_t"] = cam_t

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Dyn-HaMR npz predictions onto RGB frames")
    parser.add_argument("npz_path", type=str, help="Path to the optimization result npz file")
    parser.add_argument("images_dir", type=str, help="Directory with the original RGB frames")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for the rendered files (defaults to <npz_stem> in the images_dir)",
    )
    parser.add_argument(
        "--render-views",
        nargs="*",
        default=["src_cam"],
        choices=["src_cam", "front", "above", "side"],
        help="Camera views to render",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--vis-scale", type=float, default=1.0, help="Viewer resolution scale factor")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--mano-model-path", type=str, default="_DATA/data/mano", help="Directory with MANO models")
    parser.add_argument(
        "--mano-mean-path", type=str, default="_DATA/data/mano_mean_params.npz", help="Path to MANO mean params"
    )
    parser.add_argument("--mano-gender", type=str, default="neutral", help="MANO gender (neutral/right/left)")
    parser.add_argument("--mano-num-betas", type=int, default=10, help="Number of MANO shape coefficients")
    parser.add_argument("--temporal-smooth", action="store_true", help="Enable temporal smoothing before rendering")
    parser.add_argument("--accumulate", action="store_true", help="Accumulate frames for static views")
    parser.add_argument("--render-ground", action="store_true", help="Show the analytic ground plane in non-src views")
    parser.add_argument("--intrins", nargs=4, type=float, default=None, help="Optional override for fx fy cx cy")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)

    npz_path = Path(args.npz_path)
    img_dir = Path(args.images_dir)
    result_np = _build_result_dict(npz_path)

    B, T = result_np["trans"].shape[:2]
    print(f"Loaded {npz_path} with {B} track(s) and {T} frames")

    intrins = _get_intrinsics(result_np, args.intrins)
    print(f"Using intrinsics: {intrins.tolist()}")

    # Prepare MANO and tensors
    pose_dict = to_torch(result_np)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    mano = MANO(
        model_path=args.mano_model_path,
        num_betas=args.mano_num_betas,
        mean_params=args.mano_mean_path,
        gender=args.mano_gender,
        use_pca=False,
        batch_size=B * T,
        pose2rot=True,
    ).to(device)

    pose_dict = move_to(pose_dict, device)
    vis_mask = torch.ones(B, T, device=device)
    track_ids = torch.arange(B, device=device)

    scene_dict = prep_result_vis(
        pose_dict,
        vis_mask,
        track_ids,
        mano,
        temporal_smooth=args.temporal_smooth,
    )

    seq_name = args.output_prefix or npz_path.stem
    seq_name = os.path.basename(seq_name)
    output_prefix = args.output_prefix or str(img_dir / seq_name)

    bg_paths = _sorted_images(img_dir, T)
    sample_img = imageio.imread(bg_paths[0])
    img_h, img_w = sample_img.shape[:2]
    img_size = (img_w, img_h)

    vis = init_viewer(
        img_size,
        intrins,
        vis_scale=args.vis_scale,
        bg_paths=bg_paths,
        fps=args.fps,
    )

    try:
        animate_scene(
            vis,
            scene_dict,
            output_prefix,
            seq_name=seq_name,
            render_views=args.render_views,
            render_bg=True,
            render_ground=args.render_ground,
            accumulate=args.accumulate,
        )
    finally:
        vis.close()
        del mano


if __name__ == "__main__":
    main(parse_args())
