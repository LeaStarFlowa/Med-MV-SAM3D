from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir


def script_path(name: str) -> str:
    return str(PROJECT_ROOT / "scripts" / name)


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def maybe_run(cmd: list[str], outputs: list[Path], overwrite: bool, dry_run: bool) -> None:
    if not overwrite and outputs and all(path.exists() for path in outputs):
        print(f"Skip existing: {', '.join(str(path) for path in outputs)}")
        return
    run_cmd(cmd, dry_run=dry_run)


def run_case(case_dir: Path, args: argparse.Namespace) -> None:
    case_id = case_dir.name
    slices_json = case_dir / "slices" / "slices.json"
    candidates_dir = case_dir / "candidates_real"
    gt = case_dir / f"gt_{args.organ}.ply"
    if not slices_json.exists():
        raise FileNotFoundError(slices_json)
    if not candidates_dir.exists():
        raise FileNotFoundError(candidates_dir)
    if not gt.exists():
        raise FileNotFoundError(gt)

    quality = case_dir / "candidate_quality_report.csv"
    maybe_run(
        [
            sys.executable,
            script_path("16_check_candidates.py"),
            "--candidates",
            str(candidates_dir),
            "--output",
            str(quality),
        ],
        [quality],
        args.overwrite,
        args.dry_run,
    )

    visual_hull = case_dir / "visual_hull.ply"
    maybe_run(
        [
            sys.executable,
            script_path("15_make_visual_hull.py"),
            "--slices-json",
            str(slices_json),
            "--output",
            str(visual_hull),
            "--grid-size",
            str(args.grid_size),
            "--min-planes",
            str(args.visual_hull_min_planes),
            "--refine",
        ],
        [visual_hull],
        args.overwrite,
        args.dry_run,
    )

    top1_union = case_dir / "fused_top1_union_real.ply"
    maybe_run(
        [
            sys.executable,
            script_path("05_fuse_candidates.py"),
            "--candidates",
            str(candidates_dir),
            "--slices-json",
            str(slices_json),
            "--method",
            "union",
            "--output",
            str(top1_union),
            "--target-points",
            str(args.target_points),
            "--max-rank-per-plane",
            "1",
        ],
        [top1_union],
        args.overwrite,
        args.dry_run,
    )

    top1_silhouette = case_dir / "fused_top1_silhouette_real.ply"
    maybe_run(
        [
            sys.executable,
            script_path("05_fuse_candidates.py"),
            "--candidates",
            str(candidates_dir),
            "--slices-json",
            str(slices_json),
            "--method",
            "silhouette",
            "--output",
            str(top1_silhouette),
            "--target-points",
            str(args.target_points),
            "--max-rank-per-plane",
            "1",
        ],
        [top1_silhouette],
        args.overwrite,
        args.dry_run,
    )

    metrics_dir = ensure_dir(case_dir / "metrics_enhanced")
    evals = [
        ("visual_hull", visual_hull, metrics_dir / "metrics_visual_hull.csv"),
        ("top1_union_fusion_real", top1_union, metrics_dir / "metrics_top1_union.csv"),
        ("top1_silhouette_fusion_real", top1_silhouette, metrics_dir / "metrics_top1_silhouette.csv"),
    ]
    for method, pred, output in evals:
        maybe_run(
            [
                sys.executable,
                script_path("07_evaluate_metrics.py"),
                "--pred",
                str(pred),
                "--gt",
                str(gt),
                "--output",
                str(output),
                "--case-id",
                case_id,
                "--dataset",
                args.dataset,
                "--organ",
                args.organ,
                "--method",
                method,
                "--grid-size",
                str(args.grid_size),
            ],
            [output],
            args.overwrite,
            args.dry_run,
        )

    summary = case_dir / "summary_enhanced.csv"
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
        args.overwrite,
        args.dry_run,
    )


def aggregate(case_dirs: list[Path], args: argparse.Namespace) -> None:
    all_metrics = ensure_dir(Path(args.experiments_dir) / f"all_enhanced_metrics_{args.summary_suffix}")
    if not args.dry_run:
        for case_dir in case_dirs:
            src = case_dir / "summary_enhanced.csv"
            if src.exists():
                dst = all_metrics / f"{case_dir.name}_enhanced.csv"
                if args.overwrite or not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    summary = Path(args.experiments_dir) / f"summary_enhanced_{args.summary_suffix}.csv"
    maybe_run(
        [
            sys.executable,
            script_path("08_make_tables.py"),
            "--metrics-dir",
            str(all_metrics),
            "--output",
            str(summary),
        ],
        [summary],
        args.overwrite,
        args.dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run visual hull and top-1 fusion enhancements for existing case folders.")
    parser.add_argument("--experiments-dir", default="scripts/data/experiments/btcv_liver")
    parser.add_argument("--cases", nargs="*", help="Optional case names. Defaults to all case*/ folders.")
    parser.add_argument("--dataset", default="BTCV")
    parser.add_argument("--organ", default="liver")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--visual-hull-min-planes", type=int, default=2)
    parser.add_argument("--target-points", type=int, default=20000)
    parser.add_argument("--summary-suffix", default="20cases")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.experiments_dir)
    case_dirs = [root / name for name in args.cases] if args.cases else sorted(root.glob("case*/"))
    print(f"Running enhancements for {len(case_dirs)} cases.")
    for case_dir in case_dirs:
        print(f"\n=== {case_dir.name} ===")
        run_case(case_dir, args)
    aggregate(case_dirs, args)
    print("\nEnhancement batch complete.")


if __name__ == "__main__":
    main()
