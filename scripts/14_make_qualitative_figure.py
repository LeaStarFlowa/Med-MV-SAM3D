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


COLUMN_TITLES = ["Input slice", "Best single", "Union fusion", "Silhouette fusion", "Ground truth"]


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


def point_projection(points: np.ndarray, size: int = 256, axes: tuple[int, int] = (0, 1)) -> Image.Image:
    image = np.full((size, size), 255, dtype=np.uint8)
    if len(points) == 0:
        return Image.fromarray(image)
    xy = points[:, axes]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return Image.fromarray(image)
    xy = (xy - xy.min(axis=0, keepdims=True)) / np.maximum(np.ptp(xy, axis=0, keepdims=True), 1e-6)
    ij = np.clip(np.round(xy * (size - 1)).astype(int), 0, size - 1)
    image[ij[:, 1], ij[:, 0]] = 0
    return Image.fromarray(image).convert("RGB")


def load_projection(path: Path, size: int) -> Image.Image:
    return point_projection(load_point_cloud_ply(path), size=size)


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
    parser.add_argument("--cell-size", type=int, default=240)
    parser.add_argument("--label-height", type=int, default=34)
    parser.add_argument("--gap", type=int, default=12)
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    best_map = load_best_map(Path(args.best_single))
    cell = args.cell_size
    gap = args.gap
    label_h = args.label_height
    width = len(COLUMN_TITLES) * cell + (len(COLUMN_TITLES) + 1) * gap
    height = (len(args.cases) + 1) * (cell + label_h) + (len(args.cases) + 2) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = gap
    for col, title in enumerate(COLUMN_TITLES):
        x = gap + col * (cell + gap)
        draw_label(draw, (x + 4, y + 6), title)

    for row_idx, case_name in enumerate(args.cases):
        case_dir = experiments_dir / case_name
        if case_name not in best_map:
            raise KeyError(f"{case_name} not found in {args.best_single}")
        best = best_map[case_name]
        selected_slice = best["selected_slice"]
        candidate_name = selected_slice.replace("single_", "") + ".ply"
        images = [
            load_input_image(find_input_slice(case_dir, selected_slice), cell),
            load_projection(case_dir / "candidates_real" / candidate_name, cell),
            load_projection(case_dir / "fused_union_real.ply", cell),
            load_projection(case_dir / "fused_silhouette_real.ply", cell),
            load_projection(case_dir / "gt_liver.ply", cell),
        ]
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

