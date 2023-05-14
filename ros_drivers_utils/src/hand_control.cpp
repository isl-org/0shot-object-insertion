#include "softgrasp_ros/hand_control.h"
#include "softgrasp_ros/SetGraspParams.h"
#include "softgrasp_ros/SetGraspState.h"
#include <sstream>

HandController::HandController(const ros::NodeHandlePtr &nh_)
    : nh(nh_),
      ROS_NAME("HandController"),
      state_service_name("grasp_state"),
      params_service_name("grasp_params"),
      init_done(false),
      state_client(
          nh->serviceClient<softgrasp_ros::SetGraspState>(state_service_name)),
      params_client(nh->serviceClient<softgrasp_ros::SetGraspParams>(
          params_service_name)) {}

bool HandController::init() {
  ROS_INFO_NAMED(ROS_NAME, "Waiting for grasp state service...");
  state_client.waitForExistence();
  ROS_INFO_NAMED(ROS_NAME, "Waiting for grasp params service...");
  params_client.waitForExistence();
  init_done = true;
  init_done = set_params(0.5, 0.05);
  return init_done;
}

bool HandController::set_params(float grip_strength, float opening_amount) {
  if (!init_done) {
    ROS_ERROR_NAMED(ROS_NAME, "not initialized");
    return false;
  }
  ROS_INFO_STREAM_NAMED(ROS_NAME, "set_params (" << grip_strength << ", "
                                                 << opening_amount << ")");
  set_state(true);
  softgrasp_ros::SetGraspParams srv;
  srv.request.grip_strength = grip_strength;
  opening_amount = 4.82f*opening_amount + 0.098f;
  opening_amount = std::fmin(std::fmax(0.f, opening_amount), 1.f);
  srv.request.opening_amount = opening_amount;
  if (!params_client.call(srv)) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
                           "Could not call service " << params_service_name);
    return false;
  }
  ROS_INFO_NAMED(ROS_NAME, "%s", srv.response.message.c_str());
  set_state(false);
  return srv.response.success;
}

bool HandController::set_state(bool state) {
  if (!init_done) {
    ROS_ERROR_NAMED(ROS_NAME, "not initialized");
    return false;
  }
  ROS_INFO_STREAM_NAMED(ROS_NAME, "set_state " << state);
  softgrasp_ros::SetGraspState srv;
  srv.request.state = state;
  if (!state_client.call(srv)) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
                           "Could not call service " << state_service_name);
    return false;
  }
  ROS_INFO_NAMED(ROS_NAME, "%s", srv.response.message.c_str());
  ros::Duration(1.0).sleep();
  return srv.response.success;
}