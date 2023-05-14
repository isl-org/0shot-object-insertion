#pragma once
#include <ros/ros.h>
#include <string>

class ArmForceController {
 private:
  ros::NodeHandlePtr nh;
  std::string ROS_NAME, load_service_name, switch_service_name;
  ros::ServiceClient load_client, switch_client;
  ros::Publisher params_pub;
  bool init_done;
  float desired_mass, kp, ki, kd;
 public:
  std::string controller_name;
  ArmForceController(
      const ros::NodeHandlePtr &nh_,
      const std::string &controller_name_ = "downward_force_controller");
  bool init();
  bool start();
  bool stop();
};