from __future__ import annotations

import numpy as np

from .compat import ndi


def select_label(mask: np.ndarray, label: int | None = None) -> np.ndarray:
    if label is None:
        return np.asarray(mask) > 0
    return np.asarray(mask) == label


def normalize_ct(
    volume: np.ndarray,
    window_level: float | None = None,
    window_width: float | None = None,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    if window_level is not None and window_width is not None:
        low = window_level - window_width / 2.0
        high = window_level + window_width / 2.0
    else:
        low, high = np.percentile(volume[np.isfinite(volume)], percentile_clip)
    clipped = np.clip(volume, low, high)
    return ((clipped - low) / max(high - low, 1e-6)).astype(np.float32)


def normalize_mri(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    region = volume[mask > 0] if mask is not None and np.any(mask) else volume[np.isfinite(volume)]
    mean = float(np.mean(region))
    std = float(np.std(region))
    z = (volume - mean) / max(std, 1e-6)
    z = np.clip(z, -3.0, 3.0)
    return ((z + 3.0) / 6.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image)
    if image.max() > 1.0 or image.min() < 0.0:
        low, high = np.percentile(image, [1, 99])
        image = (np.clip(image, low, high) - low) / max(high - low, 1e-6)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask) > 0
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.uint8)
    eroded = ndi.binary_erosion(mask)
    return (mask ^ eroded).astype(np.uint8)


def distance_inside_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask) > 0
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.uint8)
    dist = ndi.distance_transform_edt(mask)
    dist = dist / max(float(dist.max()), 1e-6)
    return to_uint8(dist)


def pseudo_rgb(slice_image: np.ndarray, mask: np.ndarray, mode: str = "mask_boundary_distance") -> np.ndarray:
    gray = to_uint8(slice_image)
    mask_bool = np.asarray(mask) > 0
    if mode == "gray":
        return np.stack([gray, gray, gray], axis=-1)
    if mode == "masked_gray":
        masked = gray * mask_bool.astype(np.uint8)
        return np.stack([masked, gray, to_uint8(mask_bool.astype(np.float32))], axis=-1)
    if mode == "mask_boundary_distance":
        masked = gray * mask_bool.astype(np.uint8)
        boundary = mask_boundary(mask_bool) * 255
        distance = distance_inside_mask(mask_bool)
        return np.stack([masked, boundary, distance], axis=-1)
    raise ValueError(f"Unknown pseudo-RGB mode: {mode}")


def extract_surface_points(
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_points: int | None = 20000,
    seed: int = 0,
) -> np.ndarray:
    mask = np.asarray(mask) > 0
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)
    eroded = ndi.binary_erosion(mask)
    boundary = mask & ~eroded
    xyz = np.argwhere(boundary).astype(np.float32) * np.asarray(spacing, dtype=np.float32)
    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        xyz = xyz[rng.choice(len(xyz), size=max_points, replace=False)]
    return xyz.astype(np.float32)
