from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medmvsam3d.io_utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics CSV files into summary tables.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files = sorted(Path(args.metrics_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.metrics_dir}")
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    summary = df.groupby(["dataset", "organ", "method"], as_index=False).mean(numeric_only=True)
    output = Path(args.output)
    ensure_dir(output.parent)
    summary.to_csv(output, index=False)
    print(f"Wrote summary table to {output}")


if __name__ == "__main__":
    main()

