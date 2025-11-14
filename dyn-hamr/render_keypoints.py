#!/usr/bin/env python3
"""Overlay MANO keypoints and skeletons recovered from `.npz` files onto RGB frames."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from body_model import MANO, run_mano
from util.tensor import move_to, to_torch

from render_npz import _build_result_dict, _get_intrinsics, _sorted_images
from vis.tools import smooth_results

HAND_SKELETON: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

RIGHT_COLOR = (50, 180, 255)
LEFT_COLOR = (255, 90, 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=str, help="Path to the optimization result npz file")
    parser.add_argument("images_dir", type=str, help="Directory containing the original RGB frames")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for annotated frames (default: <images_dir>/<npz_stem>_kpts)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS for optional videos")
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Optional path to store an mp4 of the annotated frames",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--intrins",
        nargs=4,
        type=float,
        default=None,
        metavar=("fx", "fy", "cx", "cy"),
        help="Optional override for the pinhole intrinsics",
    )
    parser.add_argument("--circle-radius", type=int, default=5, help="Radius (in px) for keypoint markers")
    parser.add_argument("--line-thickness", type=int, default=2, help="Thickness (in px) for bones")
    parser.add_argument("--alpha", type=float, default=0.85, help="Blend factor between original image and overlays")
    parser.add_argument(
        "--mano-model-path",
        type=str,
        default="_DATA/data/mano",
        help="Directory with MANO models",
    )
    parser.add_argument(
        "--mano-mean-path",
        type=str,
        default="_DATA/data/mano_mean_params.npz",
        help="Path to MANO mean params",
    )
    parser.add_argument("--mano-gender", type=str, default="neutral", help="MANO gender (neutral/right/left)")
    parser.add_argument("--mano-num-betas", type=int, default=10, help="Number of MANO shape coefficients")
    parser.add_argument("--temporal-smooth", action="store_true", help="Smooth pose sequence before drawing")
    return parser.parse_args()


def _project_joints(
    joints: torch.Tensor,
    cam_R: torch.Tensor,
    cam_t: torch.Tensor,
    intrins: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world-frame joints into image coordinates."""
    B, T, J, _ = joints.shape
    if cam_R.shape[0] != T or cam_t.shape[0] != T:
        raise ValueError(
            f"Camera extrinsics must have {T} frames, got {cam_R.shape[0]} rotations and {cam_t.shape[0]} translations"
        )

    fx, fy, cx, cy = intrins
    cam_R = cam_R.to(joints.device)
    cam_t = cam_t.to(joints.device)
    points_cam = torch.einsum("tij,btnj->btni", cam_R, joints)
    points_cam = points_cam + cam_t[None, :, None, :]
    depth = points_cam[..., 2]
    points_xy = points_cam[..., :2] / depth.unsqueeze(-1).clamp(min=1e-6)
    proj = torch.empty_like(points_xy)
    proj[..., 0] = fx * points_xy[..., 0] + cx
    proj[..., 1] = fy * points_xy[..., 1] + cy
    return proj, depth


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 0.999:
        return overlay
    return (alpha * overlay + (1.0 - alpha) * base).astype(np.uint8)


def _init_video_writer(path: Path, fps: int):
    writer = None
    try:
        writer = imageio.get_writer(str(path), fps=fps)
    except TypeError:
        writer = imageio.get_writer(str(path))
    return writer


def _draw_hand(
    image: np.ndarray,
    keypoints: np.ndarray,
    valid: np.ndarray,
    color: Tuple[int, int, int],
    circle_radius: int,
    line_thickness: int,
) -> None:
    pts = keypoints.astype(np.int32)
    for i, j in HAND_SKELETON:
        if valid[i] and valid[j]:
            cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, thickness=line_thickness, lineType=cv2.LINE_AA)
    for idx, is_valid in enumerate(valid):
        if is_valid:
            cv2.circle(image, tuple(pts[idx]), circle_radius, color, thickness=-1, lineType=cv2.LINE_AA)


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)

    npz_path = Path(args.npz_path)
    img_dir = Path(args.images_dir)
    result_np = _build_result_dict(npz_path)
    if "cam_R" not in result_np or "cam_t" not in result_np:
        raise KeyError("The provided npz file does not contain cam_R/cam_t needed for 2D rendering")

    intrins = torch.tensor(_get_intrinsics(result_np, args.intrins), dtype=torch.float32)
    print(f"Using intrinsics: {intrins.tolist()}")

    pose_dict = to_torch(result_np)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    pose_dict = move_to(pose_dict, device)

    B, T = pose_dict["trans"].shape[:2]
    if args.temporal_smooth:
        if "betas" not in pose_dict:
            pose_dict["betas"] = torch.zeros(B, args.mano_num_betas, device=device)
        (
            pose_dict["root_orient"],
            pose_dict["pose_body"],
            pose_dict["betas"],
            pose_dict["trans"],
        ) = smooth_results(
            pose_dict["root_orient"],
            pose_dict["pose_body"],
            pose_dict["betas"],
            pose_dict["is_right"],
            pose_dict["trans"],
        )

    mano = MANO(
        model_path=args.mano_model_path,
        num_betas=args.mano_num_betas,
        mean_params=args.mano_mean_path,
        gender=args.mano_gender,
        use_pca=False,
        batch_size=B * T,
        pose2rot=True,
    ).to(device)

    with torch.no_grad():
        mano_out = run_mano(
            mano,
            pose_dict["trans"],
            pose_dict["root_orient"],
            pose_dict["pose_body"],
            pose_dict["is_right"],
            pose_dict.get("betas"),
        )
    joints = mano_out["joints"].to(device)
    is_right = pose_dict["is_right"].detach().cpu().numpy() > 0.5

    cam_R = torch.tensor(result_np["cam_R"], dtype=torch.float32, device=device)
    cam_t = torch.tensor(result_np["cam_t"], dtype=torch.float32, device=device)
    proj_2d, depth = _project_joints(joints, cam_R, cam_t, intrins.to(device))
    proj_2d = proj_2d.cpu().numpy()
    depth = depth.cpu().numpy()

    bg_paths = _sorted_images(img_dir, T)
    sample_img = imageio.imread(bg_paths[0])
    img_h, img_w = sample_img.shape[:2]

    seq_name = npz_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else img_dir / f"{seq_name}_kpts"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None
    if args.save_video:
        video_path = Path(args.save_video)
        if not video_path.suffix:
            video_path = video_path.with_suffix(".mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = _init_video_writer(video_path, args.fps)
        print(f"Writing video to {video_path}")

    for frame_idx in range(T):
        frame = imageio.imread(bg_paths[frame_idx])
        if frame.shape[0] != img_h or frame.shape[1] != img_w:
            frame = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        overlay = frame.copy()
        for track_idx in range(B):
            keypoints = proj_2d[track_idx, frame_idx]
            valid = depth[track_idx, frame_idx] > 1e-4
            if not np.any(valid):
                continue
            color = RIGHT_COLOR if is_right[track_idx, frame_idx] else LEFT_COLOR
            _draw_hand(
                overlay,
                keypoints,
                valid,
                color,
                circle_radius=args.circle_radius,
                line_thickness=args.line_thickness,
            )
        blended = _blend(frame, overlay, args.alpha)
        out_path = output_dir / f"{frame_idx:06d}.png"
        imageio.imwrite(out_path, blended)
        if video_writer is not None:
            video_writer.append_data(blended)

    if video_writer is not None:
        video_writer.close()
        print("Finished writing video")

    vis_name = args.save_video or str(output_dir)
    print(f"Annotated frames saved under {output_dir} (video: {vis_name})")


if __name__ == "__main__":
    main(parse_args())
