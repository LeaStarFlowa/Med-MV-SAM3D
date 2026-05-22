from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir


HIGHER_IS_BETTER = {"dice", "voxel_iou", "voxel_dice", "f1_001"}
LOWER_IS_BETTER = {"cd", "emd", "hd95", "asd", "volume_error"}
METRICS = ["dice", "voxel_iou", "voxel_dice", "f1_001", "cd", "emd", "hd95", "asd", "volume_error"]

METHOD_NAMES = {
    "union_fusion_real": "Union fusion",
    "silhouette_fusion_real": "Silhouette fusion",
    "weighted_fusion_real": "Weighted fusion",
    "med_mv_sam3d_real": "Anatomical refinement",
    "visual_hull": "Visual hull",
    "top1_union_fusion_real": "Top-1 union fusion",
    "top1_silhouette_fusion_real": "Top-1 silhouette fusion",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_wins(candidate: float, baseline: float, metric: str) -> bool:
    if metric in HIGHER_IS_BETTER:
        return candidate >= baseline
    if metric in LOWER_IS_BETTER:
        return candidate <= baseline
    raise ValueError(f"Unknown metric direction for {metric}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-metric win rates against best-single SAM3D.")
    parser.add_argument("--experiments-dir", required=True, help="Directory containing case*/summary_fusion.csv")
    parser.add_argument("--best-single", required=True, help="best_single_Ncases.csv")
    parser.add_argument("--output", required=True, help="Output win_rate_Ncases.csv")
    parser.add_argument(
        "--summary-files",
        nargs="+",
        default=["summary_fusion.csv"],
        help="Per-case summary CSV names to merge, e.g. summary_fusion.csv summary_enhanced.csv",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    best_rows = read_rows(Path(args.best_single))
    methods = list(METHOD_NAMES)
    wins = {method: {metric: 0 for metric in METRICS} for method in methods}
    valid_cases = {method: 0 for method in methods}
    case_score_sum = {method: 0 for method in methods}

    for best in best_rows:
        case_dir = experiments_dir / best["case_dir"]
        fusion_rows = {}
        for summary_name in args.summary_files:
            summary_path = case_dir / summary_name
            if not summary_path.exists():
                print(f"Warning: missing {summary_path}")
                continue
            fusion_rows.update({row["method"]: row for row in read_rows(summary_path)})
        for method in methods:
            if method not in fusion_rows:
                continue
            valid_cases[method] += 1
            row = fusion_rows[method]
            case_wins = 0
            for metric in METRICS:
                if metric_wins(float(row[metric]), float(best[metric]), metric):
                    wins[method][metric] += 1
                    case_wins += 1
            case_score_sum[method] += case_wins

    output_rows: list[dict[str, str]] = []
    for method in methods:
        n = valid_cases[method]
        row = {
            "method": METHOD_NAMES[method],
            "method_key": method,
            "num_cases": str(n),
            "mean_metric_wins_per_case": f"{case_score_sum[method] / n:.4f}" if n else "nan",
        }
        for metric in METRICS:
            count = wins[method][metric]
            row[f"{metric}_wins"] = f"{count}/{n}"
            row[f"{metric}_win_rate"] = f"{count / n:.4f}" if n else "nan"
        output_rows.append(row)

    output = Path(args.output)
    ensure_dir(output.parent)
    fieldnames = [
        "method",
        "method_key",
        "num_cases",
        "mean_metric_wins_per_case",
        *[item for metric in METRICS for item in (f"{metric}_wins", f"{metric}_win_rate")],
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote win-rate table to {output}")


if __name__ == "__main__":
    main()
