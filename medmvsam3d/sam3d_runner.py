from __future__ import annotations

import os
import subprocess

from pathlib import Path

import numpy as np

from .io_utils import ensure_dir, load_mask_png, save_point_cloud_ply, load_point_cloud_ply


def _sample_ellipsoid_from_mask(mask: np.ndarray, n_points: int = 10000, seed: int = 0) -> np.ndarray:
    mask = np.asarray(mask) > 0
    rng = np.random.default_rng(seed)
    if not np.any(mask):
        return rng.normal(size=(n_points, 3)).astype(np.float32) * 0.05

    ys, xs = np.where(mask)
    h, w = mask.shape
    cx = (xs.mean() / max(w - 1, 1)) * 2.0 - 1.0
    cy = (ys.mean() / max(h - 1, 1)) * 2.0 - 1.0
    rx = max((xs.max() - xs.min() + 1) / max(w, 1), 0.05)
    ry = max((ys.max() - ys.min() + 1) / max(h, 1), 0.05)
    rz = float(np.sqrt(rx * ry))

    phi = rng.uniform(0, 2 * np.pi, size=n_points)
    costheta = rng.uniform(-1, 1, size=n_points)
    theta = np.arccos(costheta)
    x = rx * np.sin(theta) * np.cos(phi) + cx
    y = ry * np.sin(theta) * np.sin(phi) + cy
    z = rz * np.cos(theta)
    points = np.stack([x, y, z], axis=1)
    noise = rng.normal(scale=0.015, size=points.shape)
    return np.clip(points + noise, -1.0, 1.0).astype(np.float32)


def run_sam3d_stub(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    n_points: int = 10000,
    seed: int = 0,
) -> np.ndarray:
    """Simulated SAM3D output used until the real SAM3D runtime is wired in."""
    del image_path
    mask = load_mask_png(mask_path)
    points = _sample_ellipsoid_from_mask(mask, n_points=n_points, seed=seed)
    save_point_cloud_ply(points, output_path)
    return points


def run_sam3d(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    n_points: int = 10000,
    use_stub: bool = True,
    seed: int = 0,
) -> np.ndarray:
    ensure_dir(Path(output_path).parent)

    if use_stub:
        return run_sam3d_stub(image_path, mask_path, output_path, n_points=n_points, seed=seed)
    
    project_root = Path(__file__).resolve().parents[1]
    wrapper = project_root / "scripts" / "sam3d_infer_one.py"
    python_bin = os.environ.get("SAM3D_PYTHON", "python")

    cmd = [
        python_bin,
        str(wrapper),
        "--image",
        str(image_path),
        "--mask",
        str(mask_path),
        "--output",
        str(output_path),
        "--seed",
        str(seed),
    ]

    sam3d_repo = os.environ.get("SAM3D_REPO")
    sam3d_config = os.environ.get("SAM3D_CONFIG")
    if sam3d_repo:
        cmd += ["--sam3d-repo", sam3d_repo]
    if sam3d_config:
        cmd += ["--config", sam3d_config]

    subprocess.run(cmd, check=True)

    return load_point_cloud_ply(output_path)

