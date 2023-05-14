#pragma once
#include <ros/ros.h>
#include <string>

class HandController {
 private:
  ros::NodeHandlePtr nh;
  std::string ROS_NAME, state_service_name, params_service_name;
  ros::ServiceClient state_client, params_client;
  bool init_done;
 public:
  HandController(const ros::NodeHandlePtr &nh_);
  bool init();
  bool set_params(float grip_strength, float opening_amount);
  bool set_state(bool s);
};