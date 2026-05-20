from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


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
    data = path.read_bytes()
    header_end = data.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"{path} is not a valid PLY file: missing end_header.")

    body_start = data.find(b"\n", header_end)
    if body_start < 0:
        raise ValueError(f"{path} is not a valid PLY file: truncated header.")
    body_start += 1

    header = data[:body_start].decode("latin1").splitlines()
    fmt = "ascii"
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None

    for line in header:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) >= 3 and parts[1] != "list":
                vertex_properties.append((parts[2], parts[1]))

    if vertex_count == 0:
        return np.empty((0, 3), dtype=np.float32)
    property_names = [name for name, _ in vertex_properties]
    for required in ("x", "y", "z"):
        if required not in property_names:
            raise ValueError(f"{path} has no vertex property '{required}'.")

    if fmt == "ascii":
        text = data[body_start:].decode("latin1")
        rows = [line.split() for line in text.splitlines() if line.strip()]
        if not rows:
            return np.empty((0, 3), dtype=np.float32)
        arr = np.asarray(rows[:vertex_count], dtype=np.float32)
        x_idx, y_idx, z_idx = (property_names.index(name) for name in ("x", "y", "z"))
        return arr[:, [x_idx, y_idx, z_idx]].astype(np.float32)

    if fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format in {path}: {fmt}")

    endian = "<" if fmt == "binary_little_endian" else ">"
    dtype_fields = []
    for name, ply_type in vertex_properties:
        if ply_type not in PLY_DTYPE_MAP:
            raise ValueError(f"Unsupported PLY property type in {path}: {ply_type}")
        dtype_fields.append((name, endian + PLY_DTYPE_MAP[ply_type]))
    dtype = np.dtype(dtype_fields)
    vertices = np.frombuffer(data, dtype=dtype, count=vertex_count, offset=body_start)
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    return np.asarray(points, dtype=np.float32)
