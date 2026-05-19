from __future__ import annotations

import numpy as np

from .compat import ndi


def largest_connected_component(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid) > 0
    labels, count = ndi.label(grid)
    if count == 0:
        return grid
    sizes = ndi.sum(grid, labels, index=np.arange(1, count + 1))
    largest = int(np.argmax(sizes) + 1)
    return labels == largest


def fill_holes(grid: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(np.asarray(grid) > 0)


def binary_close(grid: np.ndarray, iterations: int = 1) -> np.ndarray:
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    return ndi.binary_closing(np.asarray(grid) > 0, structure=structure, iterations=iterations)


def smooth_voxel_grid(grid: np.ndarray, sigma: float = 0.75, threshold: float = 0.35) -> np.ndarray:
    smooth = ndi.gaussian_filter((np.asarray(grid) > 0).astype(np.float32), sigma=sigma)
    return smooth >= threshold


def volume_prior_correction(
    grid: np.ndarray,
    target_volume: float | None = None,
    tolerance: float = 0.20,
    max_iterations: int = 8,
) -> np.ndarray:
    """Scale occupancy around its centroid when volume is far from a target voxel count."""
    grid = np.asarray(grid) > 0
    if target_volume is None or target_volume <= 0 or not np.any(grid):
        return grid
    current = float(grid.sum())
    if abs(current - target_volume) / target_volume <= tolerance:
        return grid

    coords = np.argwhere(grid).astype(np.float32)
    centroid = coords.mean(axis=0, keepdims=True)
    scale = (target_volume / max(current, 1.0)) ** (1.0 / 3.0)
    out_shape = np.asarray(grid.shape)
    corrected = grid
    for _ in range(max_iterations):
        new_coords = np.round((coords - centroid) * scale + centroid).astype(int)
        valid = np.all((new_coords >= 0) & (new_coords < out_shape), axis=1)
        candidate = np.zeros_like(grid)
        candidate[tuple(new_coords[valid].T)] = True
        candidate = fill_holes(largest_connected_component(candidate))
        corrected = candidate
        current = float(corrected.sum())
        if abs(current - target_volume) / target_volume <= tolerance:
            break
        scale *= (target_volume / max(current, 1.0)) ** (1.0 / 3.0)
    return corrected


def refine_voxel_grid(
    grid: np.ndarray,
    close_iterations: int = 1,
    smooth_sigma: float = 0.75,
    target_volume: float | None = None,
) -> np.ndarray:
    refined = largest_connected_component(grid)
    refined = binary_close(refined, iterations=close_iterations)
    refined = fill_holes(refined)
    refined = smooth_voxel_grid(refined, sigma=smooth_sigma)
    refined = largest_connected_component(refined)
    refined = volume_prior_correction(refined, target_volume=target_volume)
    return refined.astype(bool)
