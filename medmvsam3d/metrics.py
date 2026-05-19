from __future__ import annotations

import numpy as np

from .compat import ndi, nearest_neighbor_distances
from .fusion import normalize_point_cloud, points_to_voxel_grid


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred) > 0
    gt = np.asarray(gt) > 0
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(pred, gt).sum()) / denom


def voxel_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred) > 0
    gt = np.asarray(gt) > 0
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(pred, gt).sum()) / union


def voxel_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return dice_coefficient(pred, gt)


def nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return nearest_neighbor_distances(a, b)


def f1_at_threshold(pred_points: np.ndarray, gt_points: np.ndarray, threshold: float = 0.01) -> float:
    pred_points = normalize_point_cloud(pred_points)
    gt_points = normalize_point_cloud(gt_points)
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 1.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0
    precision = float(np.mean(nearest_distances(pred_points, gt_points) <= threshold))
    recall = float(np.mean(nearest_distances(gt_points, pred_points) <= threshold))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    pred_points = normalize_point_cloud(pred_points)
    gt_points = normalize_point_cloud(gt_points)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")
    return float(nearest_distances(pred_points, gt_points).mean() + nearest_distances(gt_points, pred_points).mean())


def earth_movers_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_points: int = 2048,
    seed: int = 0,
) -> float:
    pred_points = normalize_point_cloud(pred_points)
    gt_points = normalize_point_cloud(gt_points)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")
    rng = np.random.default_rng(seed)
    if len(pred_points) > max_points:
        pred_points = pred_points[rng.choice(len(pred_points), size=max_points, replace=False)]
    if len(gt_points) > max_points:
        gt_points = gt_points[rng.choice(len(gt_points), size=max_points, replace=False)]
    n = min(len(pred_points), len(gt_points))
    pred_points = pred_points[:n]
    gt_points = gt_points[:n]
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        cost = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=-1)
        row, col = linear_sum_assignment(cost)
        return float(cost[row, col].mean())
    except ImportError:
        pred_sorted = pred_points[np.lexsort((pred_points[:, 2], pred_points[:, 1], pred_points[:, 0]))]
        gt_sorted = gt_points[np.lexsort((gt_points[:, 2], gt_points[:, 1], gt_points[:, 0]))]
        return float(np.linalg.norm(pred_sorted - gt_sorted, axis=1).mean())


def surface_voxels(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid) > 0
    if not np.any(grid):
        return np.empty((0, 3), dtype=np.float32)
    eroded = ndi.binary_erosion(grid)
    return np.argwhere(grid & ~eroded).astype(np.float32)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_surface = surface_voxels(pred)
    gt_surface = surface_voxels(gt)
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float("inf")
    distances = np.concatenate([nearest_distances(pred_surface, gt_surface), nearest_distances(gt_surface, pred_surface)])
    return float(np.percentile(distances, 95))


def asd(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_surface = surface_voxels(pred)
    gt_surface = surface_voxels(gt)
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float("inf")
    distances = np.concatenate([nearest_distances(pred_surface, gt_surface), nearest_distances(gt_surface, pred_surface)])
    return float(distances.mean())


def relative_volume_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_volume = float((np.asarray(pred) > 0).sum())
    gt_volume = float((np.asarray(gt) > 0).sum())
    if gt_volume == 0:
        return 0.0 if pred_volume == 0 else float("inf")
    return abs(pred_volume - gt_volume) / gt_volume


def evaluate_all(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_grid: np.ndarray | None = None,
    gt_grid: np.ndarray | None = None,
    grid_size: int = 64,
) -> dict[str, float]:
    if pred_grid is None:
        pred_grid = points_to_voxel_grid(pred_points, grid_size=grid_size)
    if gt_grid is None:
        gt_grid = points_to_voxel_grid(gt_points, grid_size=grid_size)
    return {
        "dice": dice_coefficient(pred_grid, gt_grid),
        "voxel_iou": voxel_iou(pred_grid, gt_grid),
        "voxel_dice": voxel_dice(pred_grid, gt_grid),
        "f1_001": f1_at_threshold(pred_points, gt_points, threshold=0.01),
        "cd": chamfer_distance(pred_points, gt_points),
        "emd": earth_movers_distance(pred_points, gt_points),
        "hd95": hd95(pred_grid, gt_grid),
        "asd": asd(pred_grid, gt_grid),
        "volume_error": relative_volume_error(pred_grid, gt_grid),
    }
