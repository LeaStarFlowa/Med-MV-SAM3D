from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir


METRICS = ["dice", "voxel_iou", "voxel_dice", "f1_001", "cd", "emd", "hd95", "asd", "volume_error"]

METHOD_NAMES = {
    "best_single_sam3d": "Best single-slice SAM3D",
    "union_fusion_real": "Union fusion",
    "silhouette_fusion_real": "Silhouette fusion",
    "weighted_fusion_real": "Weighted fusion",
    "med_mv_sam3d_real": "Anatomical refinement",
    "visual_hull": "Visual hull",
    "top1_union_fusion_real": "Top-1 union fusion",
    "top1_silhouette_fusion_real": "Top-1 silhouette fusion",
}

METHOD_ORDER = [
    "union_fusion_real",
    "silhouette_fusion_real",
    "weighted_fusion_real",
    "med_mv_sam3d_real",
    "visual_hull",
    "top1_union_fusion_real",
    "top1_silhouette_fusion_real",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def metric_summary(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    summary: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric, "") not in {"", "nan", "NaN", "inf", "Inf"}]
        summary[metric] = (mean(values), std(values))
    return summary


def format_value(avg: float, sd: float, include_std: bool) -> str:
    if include_std:
        return f"{avg:.4f} +/- {sd:.4f}"
    return f"{avg:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paper-style main result table from best-single and fusion summaries.")
    parser.add_argument("--best-single", required=True, help="best_single_Ncases.csv")
    parser.add_argument("--fusion-summary", required=True, help="summary_fusion_Ncases.csv")
    parser.add_argument("--extra-summary", action="append", default=[], help="Additional summary CSV to append.")
    parser.add_argument("--output", required=True, help="Output main_table_Ncases.csv")
    parser.add_argument("--include-std", action="store_true", help="Write values as mean +/- std.")
    args = parser.parse_args()

    best_rows = read_rows(Path(args.best_single))
    fusion_rows = read_rows(Path(args.fusion_summary))
    for extra in args.extra_summary:
        fusion_rows.extend(read_rows(Path(extra)))
    out_rows: list[dict[str, str]] = []

    best_summary = metric_summary(best_rows)
    out_rows.append(
        {
            "method": METHOD_NAMES["best_single_sam3d"],
            "method_key": "best_single_sam3d",
            "num_cases": str(len(best_rows)),
            **{metric: format_value(*best_summary[metric], include_std=args.include_std) for metric in METRICS},
        }
    )

    fusion_by_method = {row["method"]: row for row in fusion_rows}
    ordered_fusion_rows = [fusion_by_method[key] for key in METHOD_ORDER if key in fusion_by_method]
    ordered_fusion_rows.extend(row for row in fusion_rows if row["method"] not in METHOD_ORDER)

    for row in ordered_fusion_rows:
        method_key = row["method"]
        out_rows.append(
            {
                "method": METHOD_NAMES.get(method_key, method_key),
                "method_key": method_key,
                "num_cases": str(len(best_rows)),
                **{metric: f"{float(row[metric]):.4f}" for metric in METRICS},
            }
        )

    output = Path(args.output)
    ensure_dir(output.parent)
    fieldnames = ["method", "method_key", "num_cases", *METRICS]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote main table to {output}")


if __name__ == "__main__":
    main()
