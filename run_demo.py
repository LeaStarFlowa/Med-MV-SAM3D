from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from medmvsam3d.fusion import (
    load_plane_masks_from_records,
    points_to_voxel_grid,
    silhouette_constrained_fusion,
    union_fusion,
    voxel_grid_to_points,
    weighted_fusion,
)
from medmvsam3d.io_utils import ensure_dir, save_json, save_npz_volume, save_point_cloud_ply
from medmvsam3d.metrics import evaluate_all
from medmvsam3d.preprocessing import extract_surface_points, normalize_ct
from medmvsam3d.priors import refine_voxel_grid
from medmvsam3d.sam3d_runner import run_sam3d
from medmvsam3d.slice_sampling import extract_multiplane_slices


FIELDNAMES = [
    "case_id",
    "dataset",
    "organ",
    "method",
    "dice",
    "voxel_iou",
    "voxel_dice",
    "f1_001",
    "cd",
    "emd",
    "hd95",
    "asd",
    "volume_error",
]


def make_synthetic_case(shape: tuple[int, int, int] = (96, 96, 96)) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.indices(shape)
    cx, cy, cz = np.asarray(shape) / 2.0
    ellipsoid = ((x - cx) / 23.0) ** 2 + ((y - cy) / 17.0) ** 2 + ((z - cz) / 28.0) ** 2 <= 1.0
    notch = ((x - cx - 8) / 10.0) ** 2 + ((y - cy + 7) / 8.0) ** 2 + ((z - cz) / 14.0) ** 2 <= 1.0
    mask = ellipsoid & ~notch
    scan = np.random.default_rng(7).normal(40, 15, size=shape).astype(np.float32)
    scan += mask.astype(np.float32) * 120.0
    return normalize_ct(scan), mask


def save_quicklook(gt_points: np.ndarray, pred_points: np.ndarray, output_path: Path, max_points: int = 2500) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        def projection(points: np.ndarray, size: int = 256) -> np.ndarray:
            points = points[:max_points]
            img = np.zeros((size, size), dtype=np.uint8)
            if len(points) == 0:
                return img
            p = points[:, :2]
            p = (p - p.min(axis=0, keepdims=True)) / np.maximum(np.ptp(p, axis=0, keepdims=True), 1e-6)
            ij = np.clip(np.round(p * (size - 1)).astype(int), 0, size - 1)
            img[ij[:, 1], ij[:, 0]] = 255
            return img

        gt_img = projection(gt_points)
        pred_img = projection(pred_points)
        canvas = np.full((256, 528), 255, dtype=np.uint8)
        canvas[:, :256] = 255 - gt_img
        canvas[:, 272:] = 255 - pred_img
        Image.fromarray(canvas).save(output_path)
        return
    fig = plt.figure(figsize=(10, 5))
    for idx, (title, points) in enumerate([("Synthetic GT", gt_points[:max_points]), ("Refined", pred_points[:max_points])], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a complete Med-MV-SAM3D demo with simulated SAM3D outputs.")
    parser.add_argument("--output", default="outputs/demo_case")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--candidate-points", type=int, default=8000)
    args = parser.parse_args()

    output = ensure_dir(args.output)
    slices_dir = ensure_dir(output / "slices")
    candidates_dir = ensure_dir(output / "candidates")
    fused_dir = ensure_dir(output / "fused")
    refined_dir = ensure_dir(output / "refined")
    figures_dir = ensure_dir(output / "figures")

    scan, mask = make_synthetic_case()
    spacing = (1.0, 1.0, 1.0)
    save_npz_volume(output / "volume.npz", scan, mask, spacing)
    save_json({"case_id": "demo001", "dataset": "synthetic", "organ": "ellipsoid_organ"}, output / "metadata.json")

    gt_points = extract_surface_points(mask, spacing=spacing, max_points=20000)
    save_point_cloud_ply(gt_points, output / "gt_ellipsoid_organ.ply")

    records = extract_multiplane_slices(scan, mask, slices_dir, k=args.k)
    record_dicts = [record.__dict__ for record in records]

    point_clouds = []
    weights = []
    for i, record in enumerate(records):
        candidate_path = candidates_dir / f"{record.plane}_{record.rank:02d}.ply"
        point_cloud = run_sam3d(
            record.rgb_path,
            record.mask_path,
            candidate_path,
            n_points=args.candidate_points,
            use_stub=True,
            seed=i,
        )
        point_clouds.append(point_cloud)
        weights.append(record.area)

    union = union_fusion(point_clouds, target_points=20000)
    weighted = weighted_fusion(point_clouds, weights, target_points=20000)
    plane_masks = load_plane_masks_from_records(record_dicts)
    silhouette = silhouette_constrained_fusion(point_clouds, plane_masks, target_points=20000)

    save_point_cloud_ply(union, fused_dir / "union.ply")
    save_point_cloud_ply(weighted, fused_dir / "weighted.ply")
    save_point_cloud_ply(silhouette, fused_dir / "silhouette.ply")

    refined_grid = refine_voxel_grid(points_to_voxel_grid(silhouette, grid_size=args.grid_size), target_volume=None)
    refined_points = voxel_grid_to_points(refined_grid)
    save_point_cloud_ply(refined_points, refined_dir / "med_mv_sam3d_refined.ply")

    gt_grid = points_to_voxel_grid(gt_points, grid_size=args.grid_size)
    rows = []
    for method, points in [
        ("union_fusion", union),
        ("weighted_fusion", weighted),
        ("silhouette_fusion", silhouette),
        ("med_mv_sam3d_refined", refined_points),
    ]:
        row = {
            "case_id": "demo001",
            "dataset": "synthetic",
            "organ": "ellipsoid_organ",
            "method": method,
            **evaluate_all(points, gt_points, gt_grid=gt_grid, grid_size=args.grid_size),
        }
        rows.append(row)

    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    save_quicklook(gt_points, refined_points, figures_dir / "demo_comparison.png")
    Image.open(records[0].rgb_path).save(figures_dir / "example_pseudo_rgb.png")
    print(f"Demo complete. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
