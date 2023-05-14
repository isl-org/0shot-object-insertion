#pragma once
// provides pointcloud perception ROS services
#include <ros/ros.h>
#include <vector>
#include <open3d/geometry/PointCloud.h>
#include <tf2_ros/transform_listener.h>

#include "softgrasp_ros/PointCloudPerception.h"
#include "gpd_ros/GraspConfig.h"

class PerceptionServer {
 private:
  ros::NodeHandlePtr nh;
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  std::string ROS_NAME, grasp_message;
  bool debug_mode, found_grasp;

  std::vector<size_t> segment_tabletop_pc(
      const open3d::geometry::PointCloud &pc);
  gpd_ros::GraspConfig grasp_tabletop_pc(const open3d::geometry::PointCloud &pc);
  bool sg_service_cb(softgrasp_ros::PointCloudPerception::Request &req,
                     softgrasp_ros::PointCloudPerception::Response &res);

 public:
  PerceptionServer(const ros::NodeHandlePtr &nh_, bool debug_mode_=false);
  ros::ServiceServer sg_service;  // sg = simple grasp
};