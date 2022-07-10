import open3d as o3d
import os


"--> change the 'pcd_folder' and 'ply_folder'"
pcd_folder = "/home/benji/projects/D3Feat.pytorch/data/TUM/testing_debugging/dataset_freiburg1_xyz/"
ply_folder = "/home/benji/projects/D3Feat.pytorch/data/TUM/testing_debugging/ply/"


def pcd2ply(in_pcd, out_ply):
    """Function to convert .pcd to .ply

    Args:
        in_pcd (_type_): path to the input point cloud
        out_ply (_type_): path to the output ply file
    """    
    print("Converting {} to {}".format(in_pcd, out_ply))
    pcd = o3d.io.read_point_cloud(in_pcd)
    o3d.io.write_point_cloud(out_ply, pcd)

def pcd_folder_parser(in_pcd_folder, out_ply_folder:None):
    """Function to list all pcd files from a folder

    Args:
        in_pcd_folder (_type_): input point cloud folder
        out_ply_folder (None): output ply folder
    """    
    index = -1
    for file in sorted(os.listdir(in_pcd_folder)):
        if file.endswith(".pcd"):
            index = index + 1
            ply_file_name = "cloud_bin_{}.ply".format(index)
            pcd2ply(os.path.join(in_pcd_folder, file), os.path.join(out_ply_folder, ply_file_name))


def main():
    pcd_folder_parser(pcd_folder, ply_folder)

if __name__ == "__main__":
    main()