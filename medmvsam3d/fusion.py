from __future__ import annotations

from pathlib import Path

import numpy as np

from .compat import nearest_neighbor_distances
from .io_utils import load_mask_png


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return points
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(centered)))
    return centered / max(scale, 1e-6)


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def statistical_outlier_removal(points: np.ndarray, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) <= k:
        return points
    try:
        from scipy.spatial import cKDTree  # type: ignore

        distances, _ = cKDTree(points).query(points, k=k + 1)
        mean_dist = distances[:, 1:].mean(axis=1)
    except ImportError:
        mean_dist = nearest_neighbor_distances(points, points + 1e-6)
    threshold = float(mean_dist.mean() + std_ratio * mean_dist.std())
    return points[mean_dist <= threshold]


def random_sample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) <= n:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=n, replace=False)]


def union_fusion(point_clouds: list[np.ndarray], target_points: int = 20000) -> np.ndarray:
    valid = [normalize_point_cloud(pc) for pc in point_clouds if len(pc) > 0]
    if not valid:
        return np.empty((0, 3), dtype=np.float32)
    fused = np.concatenate(valid, axis=0)
    fused = voxel_downsample(fused, voxel_size=2.0 / 128.0)
    fused = statistical_outlier_removal(fused)
    return random_sample(fused, target_points)


def weighted_fusion(
    point_clouds: list[np.ndarray],
    weights: list[float],
    target_points: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    if len(point_clouds) != len(weights):
        raise ValueError("point_clouds and weights must have the same length.")
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = np.maximum(weights_arr, 0.0)
    if weights_arr.sum() == 0:
        weights_arr = np.ones_like(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()
    samples: list[np.ndarray] = []
    for i, (pc, weight) in enumerate(zip(point_clouds, weights_arr)):
        pc = normalize_point_cloud(pc)
        if len(pc) == 0:
            continue
        n = max(1, int(round(target_points * float(weight))))
        samples.append(random_sample(pc, min(n, len(pc)), seed=seed + i))
    return union_fusion(samples, target_points=target_points)


def points_to_voxel_grid(points: np.ndarray, grid_size: int = 64) -> np.ndarray:
    points = normalize_point_cloud(points)
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    if len(points) == 0:
        return grid
    coords = np.floor((points + 1.0) * 0.5 * (grid_size - 1)).astype(int)
    coords = np.clip(coords, 0, grid_size - 1)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid


def voxel_grid_to_points(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid > 0).astype(np.float32)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32)
    denom = np.asarray(np.array(grid.shape) - 1, dtype=np.float32)
    return coords / denom * 2.0 - 1.0


def _mask_contains(mask: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    cols = np.clip(np.round((u + 1.0) * 0.5 * (w - 1)).astype(int), 0, w - 1)
    rows = np.clip(np.round((v + 1.0) * 0.5 * (h - 1)).astype(int), 0, h - 1)
    return mask[rows, cols] > 0


def visual_hull_from_plane_masks(
    plane_masks: dict[str, np.ndarray],
    grid_size: int = 64,
    min_planes: int | None = None,
) -> np.ndarray:
    """Create a visual-hull occupancy grid from axial/coronal/sagittal silhouettes."""
    coords = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    votes = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    num_planes = 0
    if "axial" in plane_masks:
        votes += _mask_contains(plane_masks["axial"], x.ravel(), y.ravel()).reshape(votes.shape)
        num_planes += 1
    if "coronal" in plane_masks:
        votes += _mask_contains(plane_masks["coronal"], x.ravel(), z.ravel()).reshape(votes.shape)
        num_planes += 1
    if "sagittal" in plane_masks:
        votes += _mask_contains(plane_masks["sagittal"], y.ravel(), z.ravel()).reshape(votes.shape)
        num_planes += 1
    if num_planes == 0:
        return np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    threshold = min_planes if min_planes is not None else min(2, num_planes)
    return votes >= threshold


def silhouette_filter_points(points: np.ndarray, plane_masks: dict[str, np.ndarray]) -> np.ndarray:
    points = normalize_point_cloud(points)
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if "axial" in plane_masks:
        keep &= _mask_contains(plane_masks["axial"], x, y)
    if "coronal" in plane_masks:
        keep &= _mask_contains(plane_masks["coronal"], x, z)
    if "sagittal" in plane_masks:
        keep &= _mask_contains(plane_masks["sagittal"], y, z)
    return points[keep]


def silhouette_constrained_fusion(
    point_clouds: list[np.ndarray],
    plane_masks: dict[str, np.ndarray],
    target_points: int = 20000,
    min_keep_ratio: float = 0.05,
) -> np.ndarray:
    fused = union_fusion(point_clouds, target_points=target_points * 2)
    filtered = silhouette_filter_points(fused, plane_masks)
    if len(filtered) < max(1, int(len(fused) * min_keep_ratio)):
        return fused[:target_points]
    return random_sample(filtered, target_points)


def resolve_record_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    if base_dir is not None:
        candidate = Path(base_dir) / path.name
        if candidate.exists():
            return candidate
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate
    return path


def load_plane_masks_from_records(records: list[dict], base_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    best: dict[str, dict] = {}
    for record in records:
        plane = record["plane"]
        if plane not in best or int(record.get("area", 0)) > int(best[plane].get("area", 0)):
            best[plane] = record
    return {plane: load_mask_png(resolve_record_path(record["mask_path"], base_dir=base_dir)) for plane, record in best.items()}


def select_records_by_rank(records: list[dict], max_rank_per_plane: int | None = None) -> list[dict]:
    if max_rank_per_plane is None or max_rank_per_plane <= 0:
        return list(records)
    return [record for record in records if int(record.get("rank", 9999)) <= max_rank_per_plane]


def candidate_path_for_record(candidates_dir: str | Path, record: dict) -> Path:
    return Path(candidates_dir) / f"{record['plane']}_{int(record['rank']):02d}.ply"
