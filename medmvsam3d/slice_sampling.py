from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import ensure_dir, save_json, save_png
from .preprocessing import pseudo_rgb, to_uint8


PLANE_TO_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}


@dataclass(frozen=True)
class SliceRecord:
    plane: str
    index: int
    rank: int
    area: int
    image_path: str
    mask_path: str
    rgb_path: str


def get_plane_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[:, :, index]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[index, :, :]
    raise ValueError(f"Unknown plane: {plane}")


def topk_mask_slice_indices(mask: np.ndarray, plane: str, k: int = 3, min_area: int = 1) -> list[tuple[int, int]]:
    axis = PLANE_TO_AXIS[plane]
    areas = np.sum(mask > 0, axis=tuple(i for i in range(3) if i != axis))
    candidates = [(int(idx), int(area)) for idx, area in enumerate(areas) if area >= min_area]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[:k]


def extract_multiplane_slices(
    scan: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
    k: int = 3,
    pseudo_rgb_mode: str = "mask_boundary_distance",
) -> list[SliceRecord]:
    output_dir = ensure_dir(output_dir)
    records: list[SliceRecord] = []
    for plane in ("axial", "coronal", "sagittal"):
        for rank, (index, area) in enumerate(topk_mask_slice_indices(mask, plane, k=k), start=1):
            image = get_plane_slice(scan, plane, index)
            mask_slice = get_plane_slice(mask, plane, index) > 0
            prefix = f"{plane}_{rank:02d}_idx{index:04d}"
            image_path = output_dir / f"{prefix}.png"
            mask_path = output_dir / f"{prefix}_mask.png"
            rgb_path = output_dir / f"{prefix}_rgb.png"
            save_png(to_uint8(image), image_path)
            save_png(mask_slice.astype(np.uint8) * 255, mask_path)
            save_png(pseudo_rgb(image, mask_slice, mode=pseudo_rgb_mode), rgb_path)
            records.append(
                SliceRecord(
                    plane=plane,
                    index=index,
                    rank=rank,
                    area=area,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    rgb_path=str(rgb_path),
                )
            )
    save_json({"slices": [record.__dict__ for record in records]}, output_dir / "slices.json")
    return records

