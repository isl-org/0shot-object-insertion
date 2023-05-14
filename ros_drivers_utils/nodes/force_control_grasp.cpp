// sends pointcloud to the perception service, gets grasp, and executes it
// with force control
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Trigger.h>
#include <tf2_eigen/tf2_eigen.h>

#include "softgrasp_ros/arm_position_control.h"
#include "softgrasp_ros/arm_force_control.h"
#include "softgrasp_ros/geometry_utils.h"
#include "softgrasp_ros/hand_control.h"
#include "softgrasp_ros/visualization_utils.h"
#include "softgrasp_ros/PointCloudPerception.h"


using namespace ros;
using namespace std;
namespace o3dg = open3d::geometry;


class ForceControlGrasp {
 private:
  NodeHandlePtr nh;
  Subscriber cloud_sub;
  ServiceClient perception_client;
  ServiceServer grasp_service, home_service;
  sensor_msgs::PointCloud2 cloud;
  string ROS_NAME;
  bool capture_pc;
  bool live_robot;
  string perception_service_name;
  Eigen::Affine3d fTe;  // pose of end effector w.r.t. flange panda_link8
  std::unique_ptr<ArmPositionController> position_controller;
  std::unique_ptr<ArmForceController> force_controller;
  std::unique_ptr<HandController> hand_controller;
  float grip_strength;
  
  void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &msg);
  bool grasp_cb(std_srvs::Trigger::Request &req,
                std_srvs::Trigger::Response &res);
  bool home_cb(std_srvs::Trigger::Request &req,
               std_srvs::Trigger::Response &res);
 public:
  ForceControlGrasp(const NodeHandlePtr &nh_);
  // needed because some init stuff needs spinning to be running
  // https://answers.ros.org/question/302283/failed-to-fetch-current-robot-state/?answer=348497#post-id-348497
  void init_live_robot();
};

ForceControlGrasp::ForceControlGrasp(const NodeHandlePtr &nh_)
    : nh(nh_),
      capture_pc(false),
      ROS_NAME("ForceControlGrasp"),
      perception_service_name("get_simple_grasp"),
      fTe(Eigen::Affine3d::Identity()),
      cloud_sub(nh->subscribe("points", 1, &ForceControlGrasp::cloud_cb, this)),
      perception_client(nh->serviceClient<softgrasp_ros::PointCloudPerception>(
          "get_simple_grasp")),
      grasp_service(
          nh->advertiseService("grasp", &ForceControlGrasp::grasp_cb, this)),
      home_service(
          nh->advertiseService("home", &ForceControlGrasp::home_cb, this)) {
  // get the params
  if (ros::param::get("~live_robot", live_robot)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME, "Live Robot: " << live_robot);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "live_robot not set");
    ros::shutdown();
  }
  if (ros::param::get("~hand_control/grip_strength", grip_strength)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME, "Grip Strength: " << grip_strength);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "hand_control/grip_strength not set");
    ros::shutdown();
  }
  std::string force_controller_name;
  if (ros::param::get("~force_control/controller_name", force_controller_name)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME,
                          "Force controller name: " << force_controller_name);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "force_control/controller_name not set");
    ros::shutdown();
  }
  float gripper_length, gripper_z_angle;
  if (ros::param::get("~position_control/gripper_length", gripper_length)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME,
                          "Gripper Length (m): " << gripper_length);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "position_control/gripper_length not set");
    ros::shutdown();
  }
  if (ros::param::get("~position_control/gripper_z_angle", gripper_z_angle)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME,
                          "Gripper Z Angle (deg): " << gripper_z_angle);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "position_control/gripper_z_angle not set");
    ros::shutdown();
  }

  fTe.rotate(Eigen::AngleAxisd(gripper_z_angle * M_PI / 180.0,
                               Eigen::Vector3d::UnitZ()));
  fTe.translate(Eigen::Vector3d(0.0, 0.0, gripper_length));
  
  ROS_INFO_STREAM_NAMED(ROS_NAME,
                        "Waiting for service " << perception_service_name);
  perception_client.waitForExistence();

  // create the controllers
  position_controller.reset(new ArmPositionController());
  hand_controller.reset(new HandController(nh));
  force_controller.reset(new ArmForceController(nh, force_controller_name));
}

void ForceControlGrasp::init_live_robot() {
  if (!live_robot) return;
  bool done = position_controller->init();
  done &= hand_controller->init();
  done &= force_controller->init();
  if (!done) {
    ROS_ERROR_NAMED(ROS_NAME, "Error in init, shutting down");
    ros::shutdown();
  }
}

void ForceControlGrasp::cloud_cb(const sensor_msgs::PointCloud2ConstPtr &msg) {
  if (!capture_pc) return;
  cloud = *msg;
  capture_pc = false;
}

bool ForceControlGrasp::grasp_cb(std_srvs::Trigger::Request &req,
                                 std_srvs::Trigger::Response &res) {
  ROS_INFO_NAMED(ROS_NAME, "Grasp service called");
  capture_pc = true;
  ros::Duration d(0.5);
  ROS_INFO_NAMED(ROS_NAME, "Waiting for pointcloud...");
  while (cloud.width == 0) d.sleep();
  ROS_INFO_NAMED(ROS_NAME, "Got pointcloud...");

  softgrasp_ros::PointCloudPerception srv;
  srv.request.pc = cloud;
  if (!perception_client.call(srv)) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME, "Cannot call service "
                                         << perception_service_name
                                         << ", try again");
    res.success = 0;
    res.message = "Could not call perception service";
    return true;
  }
  ROS_INFO_STREAM_NAMED(ROS_NAME,
                        "Got grasp from "
                            << perception_service_name
                            << ", success = " << srv.response.success.data
                            << ", score = " << srv.response.grasp.score.data
                            << ", message = " << srv.response.message.data);
  if (srv.response.success.data == 0) {
    res.success = 0;
    res.message = "Grasp not found";
    cloud = sensor_msgs::PointCloud2();
    return true;
  }

  // show EEF
  auto grasp = srv.response.grasp;
  // pose of flange w.r.t. robot base
  Eigen::Affine3d bTe = Eigen::Affine3d::Identity(); 
  bTe.translation() = get_from_Point(grasp.position);
  bTe.linear().col(0) = get_from_Vector3(grasp.binormal);
  bTe.linear().col(1) = get_from_Vector3(grasp.axis);
  bTe.linear().col(2) = get_from_Vector3(grasp.approach);
  auto bTf = bTe * fTe.inverse();
  auto rTc = tf2::transformToEigen(srv.response.rTc);
  auto ocloud = std::make_shared<o3dg::PointCloud>(pc_ros2open3d(cloud));
  ocloud->Transform(rTc.matrix());
  o3dg::AxisAlignedBoundingBox aabb(Eigen::Vector3d(0.0, -1.0, -0.1),
                                    Eigen::Vector3d(1.0,  1.0,  2.0));
  ocloud = ocloud->Crop(aabb);
  show_with_axes({ocloud}, "panda_link8", {Eigen::Affine3d::Identity(), bTf});

  // clear the cloud
  cloud = sensor_msgs::PointCloud2();

  if (!live_robot) {
    res.success = 0;
    res.message = "live robot OFF";
    return true;
  }
  // if (!hand_controller->set_params(grip_strength, grasp.width.data+0.01f)) {
  if (!hand_controller->set_params(grip_strength, 0.15f)) {
    std::string s("Could not set grasp width");
    ROS_INFO_NAMED(ROS_NAME, "%s", s.c_str());
    res.success = 0;
    res.message = s.c_str();
    return true;
  }
  if (!position_controller->eef_pose(bTf)) {
    std::string s("Could not move arm to pre-grasp pose");
    ROS_INFO_NAMED(ROS_NAME, "%s", s.c_str());
    res.success = 0;
    res.message = s.c_str();
    return true;
  }
  if (!force_controller->start()) {
    std::string s("Could not start force controller");
    ROS_INFO_NAMED(ROS_NAME, "%s", s.c_str());
    res.success = 0;
    res.message = s.c_str();
    return true;
  }
  ros::Duration(5.0).sleep();
  if (!hand_controller->set_state(true)) {
    std::string s("Could not close grasp");
    ROS_INFO_NAMED(ROS_NAME, "%s", s.c_str());
    res.success = 0;
    res.message = s.c_str();
    return true;
  }
  ros::Duration(5.0).sleep();
  if (!force_controller->stop()) {
    std::string s("Could not stop force controller");
    ROS_INFO_NAMED(ROS_NAME, "%s", s.c_str());
    res.success = 0;
    res.message = s.c_str();
    return true;
  }
  ros::Duration(5.0).sleep();
  bTf = position_controller->get_eef();
  bTf.translation().z() += 0.2f;
  if (!position_controller->eef_pose(bTf)) {
    std::string s("Could not lift object");
    ROS_INFO_NAMED(ROS_NAME, "%s", s.c_str());
    res.success = 0;
    res.message = s.c_str();
    return true;
  }
  res.success = 255;
  res.message = "Done";
  return true;
}

bool ForceControlGrasp::home_cb(std_srvs::Trigger::Request &req,
                                std_srvs::Trigger::Response &res) {
  if (!live_robot) {
    res.success = false;
    res.message = "live robot OFF";
    return true;
  }
  ROS_INFO_NAMED(ROS_NAME, "Home service called");
  res.success = position_controller->home();
  res.message = res.success ? "Done" : "Failed";
  return true;
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "force_control_grasper");
  NodeHandlePtr nh = boost::make_shared<NodeHandle>();
  ros::AsyncSpinner spinner(4);
  spinner.start();
  
  ForceControlGrasp grasper(nh);
  grasper.init_live_robot();
  ros::waitForShutdown();
  return 0;
}