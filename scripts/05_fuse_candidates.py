from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.fusion import (
    load_plane_masks_from_records,
    silhouette_constrained_fusion,
    union_fusion,
    weighted_fusion,
)
from medmvsam3d.io_utils import load_json, load_point_cloud_ply, save_point_cloud_ply


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse candidate point clouds.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--slices-json", required=True)
    parser.add_argument("--method", choices=["union", "weighted", "silhouette"], default="silhouette")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-points", type=int, default=20000)
    args = parser.parse_args()

    records = load_json(args.slices_json)["slices"]
    candidate_files = sorted(Path(args.candidates).glob("*.ply"))
    point_clouds = [load_point_cloud_ply(file) for file in candidate_files]
    if args.method == "union":
        fused = union_fusion(point_clouds, target_points=args.target_points)
    elif args.method == "weighted":
        weights = [record["area"] for record in records[: len(point_clouds)]]
        fused = weighted_fusion(point_clouds, weights, target_points=args.target_points)
    else:
        plane_masks = load_plane_masks_from_records(records)
        fused = silhouette_constrained_fusion(point_clouds, plane_masks, target_points=args.target_points)
    save_point_cloud_ply(fused, args.output)
    print(f"Wrote fused point cloud to {args.output}")


if __name__ == "__main__":
    main()
