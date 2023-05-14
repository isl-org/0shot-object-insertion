#pragma once
#include <open3d/Open3D.h>

void show_with_axes(
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geoms,
    const std::string &window_name = "Grasp Visualization",
    const std::vector<Eigen::Affine3d> &axes_poses = {
        Eigen::Affine3d::Identity()});

void show_seg_with_axes(open3d::geometry::PointCloud pc,
                        const std::vector<size_t> idxs,
                        const std::string &window_name = "Segmentation",
                        const std::vector<Eigen::Affine3d> &axes_poses = {
                            Eigen::Affine3d::Identity()});