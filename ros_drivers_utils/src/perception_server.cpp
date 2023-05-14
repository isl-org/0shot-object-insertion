#include <geometry_msgs/TransformStamped.h>
#include <tf2_eigen/tf2_eigen.h>
#include <open3d/geometry/BoundingVolume.h>
#include <iostream>

#include "softgrasp_ros/perception_server.h"
#include "softgrasp_ros/geometry_utils.h"
#include "softgrasp_ros/visualization_utils.h"


using namespace ros;
namespace o3dg = open3d::geometry;


PerceptionServer::PerceptionServer(const NodeHandlePtr &nh_, bool debug_mode_)
    : nh(nh_),
      tf_listener(tf_buffer),
      ROS_NAME("PerceptionServer"),
      debug_mode(debug_mode_),
      found_grasp(false),
      sg_service(nh->advertiseService(
          "get_simple_grasp", &PerceptionServer::sg_service_cb, this)) {}

bool PerceptionServer::sg_service_cb(
    softgrasp_ros::PointCloudPerception::Request &req,
    softgrasp_ros::PointCloudPerception::Response &res) {
  // construct the grasp
  ROS_INFO_STREAM_NAMED(ROS_NAME, "Perception Service called");
  auto cloud = pc_ros2open3d(req.pc);

  // transform cloud to robot coordinate frame
  geometry_msgs::TransformStamped rTc_tf;
  try {
    rTc_tf =
        tf_buffer.lookupTransform("panda_link0", "camera_depth_optical_frame",
                                  ros::Time::now(), ros::Duration(3.0));
  } catch (tf2::TransformException &ex) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME, "TF Error: " << ex.what());
    res.success.data = false;
    res.message.data = ex.what();
    return true;
  }
  Eigen::Affine3d rTc = tf2::transformToEigen(rTc_tf);
  cloud.Transform(rTc.matrix());
  
  // crop
  o3dg::AxisAlignedBoundingBox aabb(Eigen::Vector3d(0.0, -1.0, -0.1),
                                    Eigen::Vector3d(0.55, 0.0, 2.0));
  cloud = *cloud.Crop(aabb);
  if (debug_mode)
    show_with_axes({std::make_shared<o3dg::PointCloud>(cloud)},
                   "cropped pointcloud");

  // compute normals
  cloud.EstimateNormals();

  // get grasp
  auto grasp = grasp_tabletop_pc(cloud);
  
  // fill out response
  res.success.data = found_grasp ? 255 : 0;
  res.message.data = grasp_message;
  res.grasp = grasp;
  res.rTc = rTc_tf;
  return true;
}

gpd_ros::GraspConfig PerceptionServer::grasp_tabletop_pc(
    const o3dg::PointCloud &cloud) {
  gpd_ros::GraspConfig grasp;

  // segment the tabletop object
  std::vector<size_t> object_idxs = segment_tabletop_pc(cloud);
  if (debug_mode) show_seg_with_axes(cloud, object_idxs, "segmented object");

  auto object_xy = cloud.SelectByIndex(object_idxs);
  auto obb = object_xy->GetOrientedBoundingBox();
  auto wTo = obb_transform(obb);
  if (debug_mode)
    show_with_axes({std::make_shared<o3dg::PointCloud>(cloud)}, "wTo",
                   {Eigen::Affine3d::Identity(), wTo});
  // ensure obb Z axis is pointing up
  Eigen::Quaterniond q;
  q.setFromTwoVectors(wTo.linear().col(2), Eigen::Vector3d::UnitZ());
  obb.Rotate(q.toRotationMatrix(), obb.center_);
  wTo = obb_transform(obb);
  if (debug_mode)
    show_with_axes({std::make_shared<o3dg::PointCloud>(cloud)}, "Rotated wTo",
                   {Eigen::Affine3d::Identity(), wTo});

  for (auto &p: object_xy->points_) p.z() = 0;  // project to XY plane
  // transform into the BB coordinates
  object_xy->Transform(wTo.inverse().matrix());
  // in BB coordinates, long->short : X->Y->Z
  auto up_normal_idxs = filter_pc_by_normal(*object_xy, Eigen::Vector3d::UnitZ());
  auto up_normal_xy = object_xy->SelectByIndex(up_normal_idxs);
  UniformRandomSampler random_sampler(-0.35 * obb.extent_.x(),
                                      0.35 * obb.extent_.x());
  size_t nn_idx, N(up_normal_xy->points_.size());
  double opening_min, opening_max;
  for (size_t t=0; t<500; t++) {
    // sample a point on the longest axis
    double axis_px = random_sampler();
    nn_idx = N;
    opening_min =  DBL_MAX;
    opening_max = -DBL_MAX;
    double min_dist(DBL_MAX);
    for (size_t i=0; i<N; i++) {
      const auto &p = up_normal_xy->points_[i];
      double dist = fabs(p.x() - axis_px);
      if (dist < 5e-3) {
        opening_max = fmax(opening_max, p.y());
        opening_min = fmin(opening_min, p.y());
        dist += fabs(p.y());
        if (dist < 5e-3 && dist < min_dist) {
          min_dist = dist;
          nn_idx = i;
        }
      }
    }
    if (nn_idx < N) break;
  }
  // nn_idx = 122;  // for cloud000.pcd
  if (nn_idx == N) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME, "Could not sample a point on the object");
    grasp_message = "Could not sample point";
    found_grasp = false;
    return grasp;
  }
  ROS_DEBUG_STREAM_NAMED(ROS_NAME, "NN idx = " << nn_idx);
  
  // fill out the grasp specification
  auto sample_p = cloud.points_[object_idxs[up_normal_idxs[nn_idx]]];
  set_Point(grasp.position, sample_p);
  // approach axis (Z) points downward
  wTo.rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
  Eigen::Vector3d approach(wTo.linear().col(2));
  set_Vector3(grasp.approach, approach);  // Z
  Eigen::Vector3d binormal(wTo.linear().col(1));
  set_Vector3(grasp.binormal, binormal);  // X
  Eigen::Vector3d axis(wTo.linear().col(0));
  set_Vector3(grasp.axis, axis);          // Y
  grasp.width.data = float(opening_max - opening_min);
  grasp.score.data = 1.0f;
  set_Point(grasp.sample, sample_p);
  found_grasp = true;
  grasp_message = "Done";

  if (debug_mode) {
    // std::cout << "binormal = " << std::endl << binormal;
    // std::cout << "hand axis = " << std::endl << axis;
    // std::cout << "approach = " << std::endl << approach;
    wTo = Eigen::Affine3d::Identity();
    wTo.translation() = sample_p;
    wTo.linear().col(0) = binormal;
    wTo.linear().col(1) = axis;
    wTo.linear().col(2) = approach;
    show_with_axes({std::make_shared<o3dg::PointCloud>(cloud)}, "grasp",
                   {wTo, Eigen::Affine3d::Identity()});
  }

  return grasp;
}

std::vector<size_t> PerceptionServer::segment_tabletop_pc(
    const o3dg::PointCloud &pc) {
  // detect tabletop plane
  Eigen::Vector4d plane;
  std::vector<size_t> plane_idxs;
  std::tie(plane, plane_idxs) = pc.SegmentPlane(0.01);

  // segment object sticking out of the plane
  // first, create an oriented BB covering the volume on top of the plane
  double max_object_height = 0.2;
  Eigen::Matrix3d nRw = Eigen::Quaterniond().setFromTwoVectors(
      plane.head(3), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  auto n_plane = pc.SelectByIndex(plane_idxs);
  n_plane->Rotate(nRw, Eigen::Vector3d::Zero());
  Eigen::Vector3d extent = n_plane->GetMaxBound() - n_plane->GetMinBound();
  extent[2] = max_object_height;
  auto p = n_plane->GetCenter();
  p[2] = n_plane->GetMaxBound().z() + 0.5*max_object_height;
  p = nRw.transpose() * p;
  o3dg::OrientedBoundingBox obb(p, nRw.transpose(), extent);
  // next, find points within this oriented BB
  return obb.GetPointIndicesWithinBoundingBox(pc.points_);
}