from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir, load_point_cloud_ply


def candidate_status(points: np.ndarray, min_points: int, max_bbox_ratio: float, min_extent: float) -> tuple[str, str]:
    if len(points) < min_points:
        return "fail", "too_few_points"
    if not np.isfinite(points).all():
        return "fail", "nan_or_inf"
    extent = np.ptp(points, axis=0)
    if float(extent.max()) < min_extent:
        return "fail", "collapsed_bbox"
    ratio = float(extent.max() / max(float(extent.min()), 1e-6))
    if ratio > max_bbox_ratio:
        return "warn", "extreme_bbox_ratio"
    return "pass", "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a candidate quality report for SAM3D .ply files.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-points", type=int, default=1000)
    parser.add_argument("--max-bbox-ratio", type=float, default=8.0)
    parser.add_argument("--min-extent", type=float, default=1e-4)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.candidates).glob("*.ply")):
        points = load_point_cloud_ply(path)
        if len(points) == 0:
            extent = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        else:
            extent = np.ptp(points, axis=0)
            centroid = points.mean(axis=0)
        status, reason = candidate_status(points, args.min_points, args.max_bbox_ratio, args.min_extent)
        rows.append(
            {
                "candidate": path.name,
                "num_points": len(points),
                "bbox_x": float(extent[0]),
                "bbox_y": float(extent[1]),
                "bbox_z": float(extent[2]),
                "bbox_ratio": float(extent.max() / max(float(extent.min()), 1e-6)) if len(points) else float("inf"),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "has_nan": bool(np.isnan(points).any()) if len(points) else False,
                "has_inf": bool(np.isinf(points).any()) if len(points) else False,
                "status": status,
                "reason": reason,
            }
        )

    output = Path(args.output)
    ensure_dir(output.parent)
    fieldnames = [
        "candidate",
        "num_points",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "bbox_ratio",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "has_nan",
        "has_inf",
        "status",
        "reason",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote candidate quality report to {output}")


if __name__ == "__main__":
    main()

