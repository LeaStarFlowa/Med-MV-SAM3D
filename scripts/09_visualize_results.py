from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from medmvsam3d.io_utils import ensure_dir, load_point_cloud_ply


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple 3D scatter comparison figure.")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-points", type=int, default=3000)
    args = parser.parse_args()

    gt = load_point_cloud_ply(args.gt)[: args.max_points]
    pred = load_point_cloud_ply(args.pred)[: args.max_points]
    fig = plt.figure(figsize=(10, 5))
    for idx, (title, points) in enumerate([("Ground truth", gt), ("Prediction", pred)], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
    ensure_dir(Path(args.output).parent)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Wrote figure to {args.output}")


if __name__ == "__main__":
    main()
