from pathlib import Path
from medmvsam3d.io_utils import load_point_cloud_ply

cand_dir = Path("scripts/data/preprocess_output/candidates_real")

for p in sorted(cand_dir.glob("*.ply")):
    pts = load_point_cloud_ply(p)
    print(p.name, pts.shape, pts.min(axis=0), pts.max(axis=0))