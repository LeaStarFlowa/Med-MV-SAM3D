import open3d as o3d
import numpy as np

path = "/data2/user_shared/xhy/Med-MV-SAM3D/scripts/data/experiments/btcv_liver/case004/refined_real.ply"
pcd = o3d.io.read_point_cloud(path)
pts = np.asarray(pcd.points)

print("points shape:", pts.shape)
print("min:", pts.min(axis=0))
print("max:", pts.max(axis=0))
print("has nan:", np.isnan(pts).any())
print("has inf:", np.isinf(pts).any())
