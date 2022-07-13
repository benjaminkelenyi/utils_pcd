from tkinter.tix import Tree
import numpy as np
import open3d as o3d


# With:
# x′=x−minxmaxx−minx
# you normalize your feature x in [0,1].

# To normalize in [−1,1] you can use:

# x′′=2x−minxmaxx−minx−1
# In general, you can always get a new variable x′′′ in [a,b]:

# x′′′=(b−a)x−minxmaxx−minx+a


pcd = o3d.io.read_point_cloud("/home/benji/projects/D3Feat.pytorch/utils/test_pcd_complete.pcd")

out_arr = np.asarray(pcd.points)
min = np.min(out_arr)
max = np.max(out_arr)
norm_pcd = 2 * (out_arr - min)/(max - min) -1

pcd_save = o3d.geometry.PointCloud()
pcd_save.points = o3d.utility.Vector3dVector(norm_pcd)
# o3d.geometry.PointCloud.estimate_normals(pcd_save)
o3d.io.write_point_cloud("test_pcd_complete_normalized.pcd", pcd_save, write_ascii=True, print_progress=True)
# o3d.visualization.draw_geometries([pcd_save])

print("normalized output array from input list : ", norm_pcd) 