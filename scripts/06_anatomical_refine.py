from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.fusion import points_to_voxel_grid, voxel_grid_to_points
from medmvsam3d.io_utils import load_point_cloud_ply, save_point_cloud_ply
from medmvsam3d.priors import refine_voxel_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply anatomical-prior voxel refinement.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--target-volume", type=float, default=None)
    args = parser.parse_args()

    points = load_point_cloud_ply(args.input)
    grid = points_to_voxel_grid(points, grid_size=args.grid_size)
    refined = refine_voxel_grid(grid, target_volume=args.target_volume)
    save_point_cloud_ply(voxel_grid_to_points(refined), args.output)
    print(f"Wrote refined point cloud to {args.output}")


if __name__ == "__main__":
    main()
