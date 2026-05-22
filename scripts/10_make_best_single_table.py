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
METRIC_FIELDS = ["dice", "voxel_iou", "voxel_dice", "f1_001", "cd", "emd", "hd95", "asd", "volume_error"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_mode(metric: str, mode: str) -> str:
    if mode != "auto":
        return mode
    if metric in HIGHER_IS_BETTER:
        return "max"
    if metric in LOWER_IS_BETTER:
        return "min"
    raise ValueError(f"Unknown metric direction for {metric}. Use --mode max or --mode min.")


def select_best_row(rows: list[dict[str, str]], metric: str, mode: str) -> dict[str, str]:
    rows = [row for row in rows if row.get(metric, "") not in {"", "nan", "NaN", "inf", "Inf"}]
    if not rows:
        raise ValueError(f"No valid rows for metric {metric}.")
    direction = metric_mode(metric, mode)
    key = lambda row: float(row[metric])
    return max(rows, key=key) if direction == "max" else min(rows, key=key)


def discover_summary_files(experiments_dir: Path, pattern: str) -> list[Path]:
    files = sorted(experiments_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No summary files found under {experiments_dir} with pattern {pattern}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one best-single-slice SAM3D row per case from each case's summary_slices.csv."
    )
    parser.add_argument("--experiments-dir", required=True, help="Directory containing case*/summary_slices.csv")
    parser.add_argument("--output", required=True, help="Output best_single_Ncases.csv")
    parser.add_argument("--summary-pattern", default="case*/summary_slices.csv")
    parser.add_argument("--select-metric", default="dice", choices=METRIC_FIELDS)
    parser.add_argument("--mode", default="auto", choices=["auto", "max", "min"])
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    summary_files = discover_summary_files(experiments_dir, args.summary_pattern)
    output_rows: list[dict[str, str]] = []

    for summary_file in summary_files:
        rows = read_csv_rows(summary_file)
        if not rows:
            continue
        best = select_best_row(rows, args.select_metric, args.mode)
        case_id = best.get("case_id") or summary_file.parent.name
        output_row = {
            "case_dir": summary_file.parent.name,
            "case_id": case_id,
            "dataset": best.get("dataset", ""),
            "organ": best.get("organ", ""),
            "selected_slice": best.get("method", ""),
            "selection_metric": args.select_metric,
        }
        for field in METRIC_FIELDS:
            output_row[field] = best.get(field, "")
        output_rows.append(output_row)

    fieldnames = ["case_dir", "case_id", "dataset", "organ", "selected_slice", "selection_metric", *METRIC_FIELDS]
    output = Path(args.output)
    ensure_dir(output.parent)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} best-single rows to {output}")


if __name__ == "__main__":
    main()

