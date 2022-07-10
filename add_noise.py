import numpy as np
import open3d as o3d
import copy

from data.TUM.testing_debugging.pcd2ply import main

__author__      = "Benjamin Kelenyi"
__copyright__   = "Copyright 2022, TUCN"


def add_occlusion_noise(data_input, radius, vis=False):
    """
        function to add occlusion noise to an exisiting point cloud

        :param p1: input point cloud
        :param p2: occlusion noise radiu
        :param p3: visualization -> True/False
        :return: geometry::PointCloud with occlusion noise
    """

    chosen_center = np.random.randint(0, len(data_input.points))

    pcd_center = data_input.points[chosen_center]
    nr_coordinates = pcd_center.shape[0]
    pcd_center = np.tile(pcd_center, len(data_input.points))
    pcd_center = pcd_center.reshape(len(data_input.points), nr_coordinates)

    points = np.asarray(data_input.points)
    points = points-pcd_center
    points = np.linalg.norm(points, axis=1)

    remaining_index = np.squeeze(np.argwhere(points >= radius))

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(data_input.points)
    pcd_o3d_remaining = o3d.geometry.PointCloud()
    pcd_o3d_remaining.points = o3d.utility.Vector3dVector(
        np.squeeze(np.asarray(data_input.points)[remaining_index]))

    if vis:
        o3d.geometry.PointCloud.estimate_normals(pcd_o3d_remaining)
        o3d.geometry.PointCloud.estimate_normals(pcd_o3d)
        pcd_o3d.paint_uniform_color([1, 0, 0])
        pcd_o3d_remaining.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([pcd_o3d_remaining+pcd_o3d])

    return pcd_o3d_remaining


def add_gaussian_noise(data_input, mu, sigma, vis=False):
    """
        function to add gaussian noise to an exisiting point cloud

        :param p1: input point cloud
        :param p2: mean
        :param p3: standard deviation
        :param p4: visualization -> True/False
        :return: geometry::PointCloud with gaussian noise
    """

    noisy_pcd = copy.deepcopy(data_input)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)

    if vis:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(noisy_pcd.points)
        pcd_o3d.paint_uniform_color([1, 0.706, 0])
        o3d.geometry.PointCloud.estimate_normals(pcd_o3d)
        o3d.visualization.draw_geometries([pcd_o3d])

    return noisy_pcd

def add_down_sampling_noise(data_input, voxel_size=0.01, vis=False):
    """_summary_

    Args:
        data_input (_type_): pcd
        voxel_size (float, optional): voxel size. Defaults to 0.01.
        vis (bool, optional): visualization. Defaults to False.

    Returns:
        _type_: voxel grid
    """    
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(data_input.points)
    pcd_o3d.paint_uniform_color([1, 0.706, 0])
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d,
                                                            voxel_size=0.05)

    o3d.geometry.PointCloud.estimate_normals(voxel_grid)


    if vis:
        o3d.visualization.draw_geometries([voxel_grid])

    return voxel_grid

def main():
    """This main function was used to test the developed functions
    """    
    data = o3d.io.read_point_cloud(
        "/home/benji/projects/D3Feat.pytorch/data/3DMatch/fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika/cloud_bin_0.ply")

    data = o3d.io.read_point_cloud("/home/benji/projects/D3Feat.pytorch/data/3DMatch/occlusion_noisy_fragments/7-scenes-redkitchen/cloud_bin_0.ply")
    o3d.geometry.PointCloud.estimate_normals(data)
    o3d.geometry.PointCloud.estimate_normals(data)
    data.paint_uniform_color([1, 0, 0])
    data.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([data])

    "occlusion noise"
    add_occlusion_noise(data, 0.40, True)

    "gaussian noise"
    add_gaussian_noise(data, 0, 0.1, True)

    "down_sampling_noise"
    add_down_sampling_noise(data, voxel_size=0.02, vis=True)


if __name__ == "__main__":
    main()