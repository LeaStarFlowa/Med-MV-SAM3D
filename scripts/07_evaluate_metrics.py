from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir, load_point_cloud_ply
from medmvsam3d.metrics import evaluate_all


FIELDNAMES = [
    "case_id",
    "dataset",
    "organ",
    "method",
    "dice",
    "voxel_iou",
    "voxel_dice",
    "f1_001",
    "cd",
    "emd",
    "hd95",
    "asd",
    "volume_error",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted .ply against ground-truth .ply.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id", default="case001")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--organ", default="organ")
    parser.add_argument("--method", default="Med-MV-SAM3D")
    parser.add_argument("--grid-size", type=int, default=64)
    args = parser.parse_args()

    metrics = evaluate_all(load_point_cloud_ply(args.pred), load_point_cloud_ply(args.gt), grid_size=args.grid_size)
    row = {
        "case_id": args.case_id,
        "dataset": args.dataset,
        "organ": args.organ,
        "method": args.method,
        **metrics,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote metrics to {output}")


if __name__ == "__main__":
    main()
