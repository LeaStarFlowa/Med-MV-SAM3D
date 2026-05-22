from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir, load_point_cloud_ply


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    scan: Path
    mask: Path
    organ: str
    label: str | None
    modality: str
    dataset: str


def run_cmd(cmd: list[str], *, cwd: Path, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def script_path(name: str) -> str:
    return str(PROJECT_ROOT / "scripts" / name)


def read_cases_csv(path: Path, defaults: argparse.Namespace) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case_id = row.get("case_id") or row.get("case") or f"case{len(cases):03d}"
            organ = row.get("organ") or defaults.organ
            label = row.get("label") if row.get("label", "") != "" else defaults.label
            modality = row.get("modality") or defaults.modality
            dataset = row.get("dataset") or defaults.dataset
            cases.append(
                CaseSpec(
                    case_id=case_id,
                    scan=Path(row["scan"]),
                    mask=Path(row["mask"]),
                    organ=organ,
                    label=label,
                    modality=modality,
                    dataset=dataset,
                )
            )
    return cases


def discover_cases_from_dirs(args: argparse.Namespace) -> list[CaseSpec]:
    scan_dir = Path(args.scan_dir)
    mask_dir = Path(args.mask_dir)
    scans = sorted(scan_dir.glob(args.scan_glob))
    masks = sorted(mask_dir.glob(args.mask_glob))
    if not scans:
        raise FileNotFoundError(f"No scans found: {scan_dir}/{args.scan_glob}")
    if not masks:
        raise FileNotFoundError(f"No masks found: {mask_dir}/{args.mask_glob}")
    if len(scans) != len(masks):
        raise ValueError(f"Found {len(scans)} scans but {len(masks)} masks. Use --cases-csv for explicit pairing.")

    cases: list[CaseSpec] = []
    for i, (scan, mask) in enumerate(zip(scans, masks)):
        case_id = f"{args.case_prefix}{i + args.case_start:03d}"
        cases.append(
            CaseSpec(
                case_id=case_id,
                scan=scan,
                mask=mask,
                organ=args.organ,
                label=args.label,
                modality=args.modality,
                dataset=args.dataset,
            )
        )
    return cases


def maybe_run(cmd: list[str], output_paths: list[Path], args: argparse.Namespace) -> None:
    if not args.overwrite and output_paths and all(path.exists() for path in output_paths):
        print(f"Skip existing: {', '.join(str(path) for path in output_paths)}")
        return
    run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)


def evaluate_single_slices(case: CaseSpec, case_dir: Path, args: argparse.Namespace) -> None:
    candidates_dir = case_dir / "candidates_real"
    metrics_dir = ensure_dir(case_dir / "metrics_slices")
    gt_path = case_dir / f"gt_{case.organ}.ply"
    candidate_files = sorted(candidates_dir.glob("*.ply"))
    if not candidate_files:
        raise FileNotFoundError(f"No candidate .ply files found in {candidates_dir}")

    for candidate in candidate_files:
        method = f"single_{candidate.stem}"
        output = metrics_dir / f"metrics_{candidate.stem}.csv"
        maybe_run(
            [
                sys.executable,
                script_path("07_evaluate_metrics.py"),
                "--pred",
                str(candidate),
                "--gt",
                str(gt_path),
                "--output",
                str(output),
                "--case-id",
                case.case_id,
                "--dataset",
                case.dataset,
                "--organ",
                case.organ,
                "--method",
                method,
                "--grid-size",
                str(args.grid_size),
            ],
            [output],
            args,
        )

    summary = case_dir / "summary_slices.csv"
    maybe_run(
        [
            sys.executable,
            script_path("08_make_tables.py"),
            "--metrics-dir",
            str(metrics_dir),
            "--output",
            str(summary),
        ],
        [summary],
        args,
    )


def check_candidates(case_dir: Path) -> None:
    candidates_dir = case_dir / "candidates_real"
    for candidate in sorted(candidates_dir.glob("*.ply")):
        points = load_point_cloud_ply(candidate)
        if len(points) == 0:
            raise ValueError(f"Empty candidate point cloud: {candidate}")


def run_case(case: CaseSpec, args: argparse.Namespace) -> None:
    case_dir = ensure_dir(Path(args.output_root) / case.case_id)
    gt_path = case_dir / f"gt_{case.organ}.ply"
    volume_path = case_dir / "volume.npz"
    slices_json = case_dir / "slices" / "slices.json"
    candidates_dir = case_dir / "candidates_real"
    fusion_metrics_dir = ensure_dir(case_dir / "metrics_fusion")
    figures_dir = ensure_dir(case_dir / "figures")

    preprocess_cmd = [
        sys.executable,
        script_path("01_preprocess_nifti.py"),
        "--scan",
        str(case.scan),
        "--mask",
        str(case.mask),
        "--case-id",
        case.case_id,
        "--organ",
        case.organ,
        "--modality",
        case.modality,
        "--window-level",
        str(args.window_level),
        "--window-width",
        str(args.window_width),
        "--output",
        str(case_dir),
    ]
    if case.label not in {None, ""}:
        preprocess_cmd += ["--label", str(case.label)]
    maybe_run(preprocess_cmd, [volume_path, gt_path], args)

    maybe_run(
        [
            sys.executable,
            script_path("02_extract_multiplane_slices.py"),
            "--processed",
            str(case_dir),
            "--k",
            str(args.k),
        ],
        [slices_json],
        args,
    )

    expected_candidates = [candidates_dir / f"{plane}_{rank:02d}.ply" for plane in ("axial", "coronal", "sagittal") for rank in range(1, args.k + 1)]
    sam3d_cmd = [
        sys.executable,
        script_path("03_run_sam3d_candidates.py"),
        "--slices-json",
        str(slices_json),
        "--output",
        str(candidates_dir),
        "--n-points",
        str(args.n_points),
    ]
    if args.real_sam3d:
        sam3d_cmd.append("--real-sam3d")
    maybe_run(sam3d_cmd, expected_candidates, args)

    if not args.dry_run:
        check_candidates(case_dir)
    evaluate_single_slices(case, case_dir, args)

    fusion_outputs = {
        "union": case_dir / "fused_union_real.ply",
        "weighted": case_dir / "fused_weighted_real.ply",
        "silhouette": case_dir / "fused_silhouette_real.ply",
    }
    for method, output in fusion_outputs.items():
        maybe_run(
            [
                sys.executable,
                script_path("05_fuse_candidates.py"),
                "--candidates",
                str(candidates_dir),
                "--slices-json",
                str(slices_json),
                "--method",
                method,
                "--output",
                str(output),
                "--target-points",
                str(args.target_points),
            ],
            [output],
            args,
        )

    refined_output = case_dir / "refined_real.ply"
    maybe_run(
        [
            sys.executable,
            script_path("06_anatomical_refine.py"),
            "--input",
            str(fusion_outputs["silhouette"]),
            "--output",
            str(refined_output),
            "--grid-size",
            str(args.grid_size),
        ],
        [refined_output],
        args,
    )

    eval_specs = [
        ("union_fusion_real", fusion_outputs["union"], fusion_metrics_dir / "metrics_union_real.csv"),
        ("weighted_fusion_real", fusion_outputs["weighted"], fusion_metrics_dir / "metrics_weighted_real.csv"),
        ("silhouette_fusion_real", fusion_outputs["silhouette"], fusion_metrics_dir / "metrics_silhouette_real.csv"),
        ("med_mv_sam3d_real", refined_output, fusion_metrics_dir / "metrics_refined_real.csv"),
    ]
    for method, pred, output in eval_specs:
        maybe_run(
            [
                sys.executable,
                script_path("07_evaluate_metrics.py"),
                "--pred",
                str(pred),
                "--gt",
                str(gt_path),
                "--output",
                str(output),
                "--case-id",
                case.case_id,
                "--dataset",
                case.dataset,
                "--organ",
                case.organ,
                "--method",
                method,
                "--grid-size",
                str(args.grid_size),
            ],
            [output],
            args,
        )

    summary_fusion = case_dir / "summary_fusion.csv"
    maybe_run(
        [
            sys.executable,
            script_path("08_make_tables.py"),
            "--metrics-dir",
            str(fusion_metrics_dir),
            "--output",
            str(summary_fusion),
        ],
        [summary_fusion],
        args,
    )

    for method, pred in [
        ("union", fusion_outputs["union"]),
        ("silhouette", fusion_outputs["silhouette"]),
        ("refined", refined_output),
    ]:
        fig = figures_dir / f"{method}_vs_gt.png"
        maybe_run(
            [
                sys.executable,
                script_path("09_visualize_results.py"),
                "--gt",
                str(gt_path),
                "--pred",
                str(pred),
                "--output",
                str(fig),
            ],
            [fig],
            args,
        )


def aggregate_outputs(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    best_output = output_root / f"best_single_{args.summary_suffix}.csv"
    maybe_run(
        [
            sys.executable,
            script_path("10_make_best_single_table.py"),
            "--experiments-dir",
            str(output_root),
            "--output",
            str(best_output),
            "--select-metric",
            args.best_metric,
        ],
        [best_output],
        args,
    )

    all_metrics = ensure_dir(output_root / f"all_fusion_metrics_{args.summary_suffix}")
    if not args.dry_run:
        for case_dir in sorted(output_root.glob("case*/")):
            src = case_dir / "summary_fusion.csv"
            if src.exists():
                dst = all_metrics / f"{case_dir.name}_fusion.csv"
                if args.overwrite or not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    fusion_summary = output_root / f"summary_fusion_{args.summary_suffix}.csv"
    maybe_run(
        [
            sys.executable,
            script_path("08_make_tables.py"),
            "--metrics-dir",
            str(all_metrics),
            "--output",
            str(fusion_summary),
        ],
        [fusion_summary],
        args,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 01-09 Med-MV-SAM3D scripts for multiple BTCV liver cases with resumable outputs."
    )
    parser.add_argument("--cases-csv", help="CSV with columns: case_id,scan,mask; optional: organ,label,modality,dataset")
    parser.add_argument("--scan-dir", help="Directory containing scan .nii.gz files, used when --cases-csv is omitted")
    parser.add_argument("--mask-dir", help="Directory containing mask .nii.gz files, used when --cases-csv is omitted")
    parser.add_argument("--scan-glob", default="*.nii.gz")
    parser.add_argument("--mask-glob", default="*.nii.gz")
    parser.add_argument("--case-prefix", default="case")
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", default="scripts/data/experiments/btcv_liver")
    parser.add_argument("--dataset", default="BTCV")
    parser.add_argument("--organ", default="liver")
    parser.add_argument("--label", default="6")
    parser.add_argument("--modality", default="CT", choices=["CT", "MRI"])
    parser.add_argument("--window-level", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--target-points", type=int, default=20000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--real-sam3d", action="store_true")
    parser.add_argument("--best-metric", default="dice")
    parser.add_argument("--summary-suffix", default="batch")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.cases_csv:
        cases = read_cases_csv(Path(args.cases_csv), args)
    else:
        if not args.scan_dir or not args.mask_dir:
            raise ValueError("Use --cases-csv or provide both --scan-dir and --mask-dir.")
        cases = discover_cases_from_dirs(args)
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"Running {len(cases)} cases. Output root: {args.output_root}")
    for case in cases:
        print(f"\n=== {case.case_id}: {case.organ} ===")
        run_case(case, args)

    aggregate_outputs(args)
    print("\nBatch complete.")


if __name__ == "__main__":
    main()

