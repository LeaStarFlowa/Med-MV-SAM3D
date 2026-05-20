from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir, read_nifti, save_json, save_npz_volume, save_point_cloud_ply
from medmvsam3d.preprocessing import extract_surface_points, normalize_ct, normalize_mri, select_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess scan/mask NIfTI into a Med-MV-SAM3D case folder.")
    parser.add_argument("--scan", required=True, help="Input scan .nii or .nii.gz")
    parser.add_argument("--mask", required=True, help="Input segmentation mask .nii or .nii.gz")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--organ", required=True)
    parser.add_argument("--label", type=int, default=None, help="Target label id. If omitted, uses mask > 0.")
    parser.add_argument("--modality", choices=["CT", "MRI"], default="CT")
    parser.add_argument("--window-level", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-gt-points", type=int, default=20000)
    args = parser.parse_args()

    output = ensure_dir(args.output)
    scan, spacing, affine = read_nifti(args.scan)
    mask_raw, mask_spacing, _ = read_nifti(args.mask)
    if scan.shape != mask_raw.shape:
        raise ValueError(f"Scan shape {scan.shape} does not match mask shape {mask_raw.shape}.")

    mask = select_label(mask_raw, args.label)
    if args.modality == "CT":
        scan_norm = normalize_ct(scan, window_level=args.window_level, window_width=args.window_width)
    else:
        scan_norm = normalize_mri(scan, mask=mask)

    save_npz_volume(output / "volume.npz", scan_norm, mask, spacing=spacing)
    gt_points = extract_surface_points(mask, spacing=spacing, max_points=args.max_gt_points)
    save_point_cloud_ply(gt_points, output / f"gt_{args.organ}.ply")
    save_json(
        {
            "case_id": args.case_id,
            "organ": args.organ,
            "label": args.label,
            "modality": args.modality,
            "scan_shape": list(scan.shape),
            "spacing": list(spacing),
            "mask_spacing": list(mask_spacing),
            "affine": affine.tolist(),
            "mask_voxels": int(mask.sum()),
        },
        output / "metadata.json",
    )
    print(f"Wrote processed case to {output}")


if __name__ == "__main__":
    main()
