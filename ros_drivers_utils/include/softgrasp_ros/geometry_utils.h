#pragma once
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/BoundingVolume.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <random>
#include <vector>

open3d::geometry::PointCloud pc_ros2open3d(
    const sensor_msgs::PointCloud2 &pc_in);

class UniformRandomSampler {
 private:
  std::random_device rd;
  std::mt19937 generator;
  std::uniform_real_distribution<double> distr;
 public:
  UniformRandomSampler(double low=0.0, double high=1.0);
  double operator()();
};

Eigen::Affine3d obb_transform(const open3d::geometry::OrientedBoundingBox &obb);

std::vector<size_t> filter_pc_by_normal(const open3d::geometry::PointCloud &pc,
                                        const Eigen::Vector3d &n,
                                        double angle_thresh_deg = 10.0);

void set_Point(geometry_msgs::Point &p, const Eigen::Vector3d &ep);
void set_Vector3(geometry_msgs::Vector3 &v, const Eigen::Vector3d &ep);
Eigen::Vector3d get_from_Point(const geometry_msgs::Point &p);
Eigen::Vector3d get_from_Vector3(const geometry_msgs::Vector3 &v);