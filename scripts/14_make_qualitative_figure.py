from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.io_utils import ensure_dir, load_point_cloud_ply


DEFAULT_COLUMNS = ["input", "best", "union", "silhouette", "visual_hull", "top1", "gt"]
COLUMN_TITLES = {
    "input": "Input slice",
    "best": "Best single",
    "union": "Union fusion",
    "silhouette": "Silhouette fusion",
    "visual_hull": "Visual hull",
    "top1": "Top-1 union",
    "refined": "Refined",
    "gt": "Ground truth",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_best_map(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
    return {row["case_dir"]: row for row in rows}


def find_input_slice(case_dir: Path, selected_slice: str) -> Path:
    # selected_slice example: single_sagittal_02
    parts = selected_slice.replace("single_", "").split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse selected slice: {selected_slice}")
    plane = parts[0]
    rank = int(parts[1])
    slices_json = case_dir / "slices" / "slices.json"
    records = json.loads(slices_json.read_text(encoding="utf-8"))["slices"]
    for record in records:
        if record["plane"] == plane and int(record["rank"]) == rank:
            candidate = Path(record["rgb_path"])
            if candidate.is_absolute():
                return candidate
            direct = case_dir.parent.parent.parent / candidate
            if direct.exists():
                return direct
            local = case_dir / "slices" / candidate.name
            if local.exists():
                return local
            return candidate
    raise FileNotFoundError(f"No slice record found for {selected_slice} in {slices_json}")


def point_projection(
    points: np.ndarray,
    size: int = 256,
    axes: tuple[int, int] = (0, 1),
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    image = np.full((size, size), 255, dtype=np.uint8)
    if len(points) == 0:
        return Image.fromarray(image)
    xy = points[:, axes]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return Image.fromarray(image)
    if bounds is None:
        lo = xy.min(axis=0, keepdims=True)
        hi = xy.max(axis=0, keepdims=True)
    else:
        lo = bounds[0][list(axes)].reshape(1, 2)
        hi = bounds[1][list(axes)].reshape(1, 2)
    xy = (xy - lo) / np.maximum(hi - lo, 1e-6)
    ij = np.clip(np.round(xy * (size - 1)).astype(int), 0, size - 1)
    image[ij[:, 1], ij[:, 0]] = 0
    return Image.fromarray(image).convert("RGB")


def load_projection(path: Path, size: int, bounds: tuple[np.ndarray, np.ndarray] | None = None) -> Image.Image:
    return point_projection(load_point_cloud_ply(path), size=size, bounds=bounds)


def finite_bounds(point_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = [points[np.isfinite(points).all(axis=1)] for points in point_sets if len(points) > 0]
    valid = [points for points in valid if len(points) > 0]
    if not valid:
        return np.array([-1, -1, -1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32)
    stacked = np.concatenate(valid, axis=0)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    pad = np.maximum((hi - lo) * 0.05, 1e-3)
    return lo - pad, hi + pad


def load_input_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill="black", font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multi-case qualitative figure for liver reconstruction.")
    parser.add_argument("--experiments-dir", required=True, help="Directory containing caseXXX folders")
    parser.add_argument("--best-single", required=True, help="best_single_Ncases.csv")
    parser.add_argument("--cases", nargs="+", required=True, help="Case directory names, e.g. case000 case001 case003")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        choices=list(COLUMN_TITLES),
        help="Columns to render.",
    )
    parser.add_argument("--organ", default="liver")
    parser.add_argument("--cell-size", type=int, default=240)
    parser.add_argument("--label-height", type=int, default=34)
    parser.add_argument("--gap", type=int, default=12)
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    best_map = load_best_map(Path(args.best_single))
    cell = args.cell_size
    gap = args.gap
    label_h = args.label_height
    columns = args.columns
    width = len(columns) * cell + (len(columns) + 1) * gap
    height = (len(args.cases) + 1) * (cell + label_h) + (len(args.cases) + 2) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = gap
    for col, column in enumerate(columns):
        x = gap + col * (cell + gap)
        draw_label(draw, (x + 4, y + 6), COLUMN_TITLES[column])

    for row_idx, case_name in enumerate(args.cases):
        case_dir = experiments_dir / case_name
        if case_name not in best_map:
            raise KeyError(f"{case_name} not found in {args.best_single}")
        best = best_map[case_name]
        selected_slice = best["selected_slice"]
        candidate_name = selected_slice.replace("single_", "") + ".ply"
        paths = {
            "best": case_dir / "candidates_real" / candidate_name,
            "union": case_dir / "fused_union_real.ply",
            "silhouette": case_dir / "fused_silhouette_real.ply",
            "visual_hull": case_dir / "visual_hull.ply",
            "top1": case_dir / "fused_top1_union_real.ply",
            "refined": case_dir / "refined_real.ply",
            "gt": case_dir / f"gt_{args.organ}.ply",
        }
        point_sets = [load_point_cloud_ply(path) for key, path in paths.items() if key in columns and path.exists()]
        bounds = finite_bounds(point_sets)
        images = []
        for column in columns:
            if column == "input":
                images.append(load_input_image(find_input_slice(case_dir, selected_slice), cell))
            else:
                path = paths[column]
                if not path.exists():
                    raise FileNotFoundError(f"Missing file for column '{column}': {path}")
                images.append(load_projection(path, cell, bounds=bounds))
        y = gap + (row_idx + 1) * (cell + label_h + gap)
        draw_label(draw, (gap, y - label_h + 6), f"{case_name} ({selected_slice})")
        for col, image in enumerate(images):
            x = gap + col * (cell + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(180, 180, 180))

    output = Path(args.output)
    ensure_dir(output.parent)
    canvas.save(output)
    print(f"Wrote qualitative figure to {output}")


if __name__ == "__main__":
    main()
