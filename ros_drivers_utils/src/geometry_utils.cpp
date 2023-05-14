#include "softgrasp_ros/geometry_utils.h"
#include <vector>
#include <Eigen/Eigen>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <iostream>

using namespace std;
namespace o3dg = open3d::geometry;


o3dg::PointCloud pc_ros2open3d(const sensor_msgs::PointCloud2 &pc_ros) {
  size_t size = pc_ros.height * pc_ros.width;
  std::vector<Eigen::Vector3d> points(size);
  sensor_msgs::PointCloud2ConstIterator<float> it(pc_ros, "x");
  for(; it!=it.end(); ++it) {
    points.emplace_back(it[0], it[1], it[2]);
  }
  return o3dg::PointCloud(points);
}

UniformRandomSampler::UniformRandomSampler(double low, double high)
    : generator(rd()), distr(low, high) {}

double UniformRandomSampler::operator()() {
  return distr(generator);
}

Eigen::Affine3d obb_transform(const o3dg::OrientedBoundingBox &obb) {
  Eigen::Affine3d wTb;
  wTb.translation() = obb.GetCenter();
  wTb.linear() = obb.R_;
  return wTb;
}

vector<size_t> filter_pc_by_normal(const o3dg::PointCloud &pc,
                                   const Eigen::Vector3d &n,
                                   double angle_thresh_deg) {
  vector<size_t> idxs;
  if (!pc.HasNormals()) {
    cerr << "filter_pc_by_normal: PC does not have normals" << endl;
    return idxs;
  }

  double c_thresh = cos(angle_thresh_deg * M_PI / 180.0);
  for (size_t i=0; i<pc.points_.size(); i++) {
    if (pc.normals_[i].dot(n) >= c_thresh) idxs.push_back(i);
  }

  return idxs;
}

void set_Point(geometry_msgs::Point &p, const Eigen::Vector3d &ep) {
  p.x = ep.x();
  p.y = ep.y();
  p.z = ep.z();
}

void set_Vector3(geometry_msgs::Vector3 &v, const Eigen::Vector3d &ep) {
  v.x = ep.x();
  v.y = ep.y();
  v.z = ep.z();
}

Eigen::Vector3d get_from_Point(const geometry_msgs::Point &p) {
  return {p.x, p.y, p.z};
}

Eigen::Vector3d get_from_Vector3(const geometry_msgs::Vector3 &v) {
  return {v.x, v.y, v.z};
}