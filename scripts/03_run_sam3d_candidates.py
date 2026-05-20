from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import load_json
from medmvsam3d.sam3d_runner import run_sam3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAM3D candidate point clouds for extracted slices.")
    parser.add_argument("--slices-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--real-sam3d", action="store_true", help="Use real SAM3D integration instead of stub.")
    args = parser.parse_args()

    records = load_json(args.slices_json)["slices"]
    output = Path(args.output)
    for i, record in enumerate(records):
        name = f"{record['plane']}_{record['rank']:02d}.ply"
        run_sam3d(
            record["rgb_path"],
            record["mask_path"],
            output / name,
            n_points=args.n_points,
            use_stub=not args.real_sam3d,
            seed=i,
        )
    print(f"Wrote {len(records)} candidate point clouds to {output}")


if __name__ == "__main__":
    main()
