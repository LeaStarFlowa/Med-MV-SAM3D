from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.fusion import load_plane_masks_from_records, visual_hull_from_plane_masks, voxel_grid_to_points
from medmvsam3d.io_utils import load_json, save_point_cloud_ply
from medmvsam3d.priors import refine_voxel_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual-hull baseline from the largest axial/coronal/sagittal masks.")
    parser.add_argument("--slices-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--min-planes", type=int, default=2, help="Voxel must satisfy at least this many plane masks.")
    parser.add_argument("--refine", action="store_true", help="Apply connected-component, closing, hole filling and smoothing.")
    args = parser.parse_args()

    records = load_json(args.slices_json)["slices"]
    plane_masks = load_plane_masks_from_records(records, base_dir=Path(args.slices_json).parent)
    grid = visual_hull_from_plane_masks(plane_masks, grid_size=args.grid_size, min_planes=args.min_planes)
    if args.refine:
        grid = refine_voxel_grid(grid, close_iterations=1, smooth_sigma=0.5)
    points = voxel_grid_to_points(grid)
    save_point_cloud_ply(points, args.output)
    print(f"Wrote visual hull with {len(points)} points to {args.output}")


if __name__ == "__main__":
    main()
