from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_nifti(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """Read a NIfTI volume as a numpy array plus voxel spacing and affine."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required to read NIfTI files. Install requirements.txt.") from exc

    image = nib.load(str(path))
    data = np.asarray(image.get_fdata())
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return data, spacing, np.asarray(image.affine)


def save_npz_volume(
    path: str | Path,
    scan: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, scan=scan.astype(np.float32), mask=mask.astype(np.uint8), spacing=np.asarray(spacing))


def load_npz_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    data = np.load(path)
    spacing = tuple(float(v) for v in data["spacing"])
    return data["scan"], data["mask"], spacing


def save_png(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_mask_png(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def save_point_cloud_ply(points: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
    )
    lines = [f"{x:.7f} {y:.7f} {z:.7f}" for x, y, z in points]
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def load_point_cloud_ply(path: str | Path) -> np.ndarray:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("end_header")
    rows = [line.split()[:3] for line in lines[end + 1 :] if line.strip()]
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

