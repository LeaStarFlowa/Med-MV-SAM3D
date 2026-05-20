from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir, load_point_cloud_ply


def save_projection_fallback(gt: np.ndarray, pred: np.ndarray, output: str | Path, size: int = 256) -> None:
    def project(points: np.ndarray) -> np.ndarray:
        img = np.zeros((size, size), dtype=np.uint8)
        if len(points) == 0:
            return img
        xy = points[:, :2]
        xy = (xy - xy.min(axis=0, keepdims=True)) / np.maximum(np.ptp(xy, axis=0, keepdims=True), 1e-6)
        ij = np.clip(np.round(xy * (size - 1)).astype(int), 0, size - 1)
        img[ij[:, 1], ij[:, 0]] = 255
        return img

    canvas = np.full((size, size * 2 + 16), 255, dtype=np.uint8)
    canvas[:, :size] = 255 - project(gt)
    canvas[:, size + 16 :] = 255 - project(pred)
    Image.fromarray(canvas).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple 3D scatter comparison figure.")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-points", type=int, default=3000)
    args = parser.parse_args()

    gt = load_point_cloud_ply(args.gt)[: args.max_points]
    pred = load_point_cloud_ply(args.pred)[: args.max_points]
    ensure_dir(Path(args.output).parent)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        save_projection_fallback(gt, pred, args.output)
        print(f"Wrote fallback projection figure to {args.output}")
        return

    fig = plt.figure(figsize=(10, 5))
    for idx, (title, points) in enumerate([("Ground truth", gt), ("Prediction", pred)], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Wrote figure to {args.output}")


if __name__ == "__main__":
    main()
