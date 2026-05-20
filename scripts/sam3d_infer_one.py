from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sam3d-repo", default=os.environ.get("SAM3D_REPO"))
    parser.add_argument("--config", default=os.environ.get("SAM3D_CONFIG", "checkpoints/hf/pipeline.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not args.sam3d_repo:
        raise ValueError("Set --sam3d-repo or SAM3D_REPO to the facebookresearch/sam-3d-objects path.")

    sam3d_repo = Path(args.sam3d_repo).resolve()
    sys.path.insert(0, str(sam3d_repo / "notebook"))

    from inference import Inference, load_image

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = sam3d_repo / config_path

    image = load_image(args.image)
    mask = np.array(Image.open(args.mask).convert("L")) > 0

    inference = Inference(str(config_path), compile=args.compile)
    output = inference(image, mask, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output["gs"].save_ply(str(output_path))


if __name__ == "__main__":
    main()