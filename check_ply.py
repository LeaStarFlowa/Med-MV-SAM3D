from pathlib import Path
from medmvsam3d.io_utils import load_point_cloud_ply

cand_dir = Path("/data2/user_shared/xhy/Med-MV-SAM3D/scripts/data/experiments/btcv_liver/case004")

for p in sorted(cand_dir.glob("*.ply")):
    pts = load_point_cloud_ply(p)
    print(p.name, pts.shape, pts.min(axis=0), pts.max(axis=0))