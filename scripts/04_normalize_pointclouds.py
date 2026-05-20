from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medmvsam3d.fusion import normalize_point_cloud
from medmvsam3d.io_utils import ensure_dir, load_point_cloud_ply, save_point_cloud_ply


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize .ply point clouds to a unit cube.")
    parser.add_argument("--input", required=True, help="Input .ply or directory of .ply files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = ensure_dir(args.output)
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.ply"))
    for file in files:
        save_point_cloud_ply(normalize_point_cloud(load_point_cloud_ply(file)), output_path / file.name)
    print(f"Normalized {len(files)} point clouds into {output_path}")


if __name__ == "__main__":
    main()
