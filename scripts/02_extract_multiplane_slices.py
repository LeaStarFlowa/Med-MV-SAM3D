from __future__ import annotations

import argparse

from medmvsam3d.io_utils import load_npz_volume
from medmvsam3d.slice_sampling import extract_multiplane_slices


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top-k axial/coronal/sagittal slices by mask area.")
    parser.add_argument("--processed", required=True, help="Processed case directory containing volume.npz")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--pseudo-rgb-mode", default="mask_boundary_distance")
    args = parser.parse_args()

    scan, mask, _ = load_npz_volume(f"{args.processed}/volume.npz")
    records = extract_multiplane_slices(
        scan,
        mask,
        f"{args.processed}/slices",
        k=args.k,
        pseudo_rgb_mode=args.pseudo_rgb_mode,
    )
    print(f"Wrote {len(records)} slices to {args.processed}/slices")


if __name__ == "__main__":
    main()

