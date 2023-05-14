#include <controller_manager_msgs/LoadController.h>
#include <controller_manager_msgs/SwitchController.h>
#include <franka_msgs/SetForceTorqueCollisionBehavior.h>
#include "softgrasp_ros/arm_force_control.h"
#include "softgrasp_ros_controllers/DownwardForceParams.h"

namespace cm_msgs = controller_manager_msgs;

ArmForceController::ArmForceController(const ros::NodeHandlePtr &nh_,
                                       const std::string &controller_name_)
    : nh(nh_),
      ROS_NAME("ArmForceController"),
      load_service_name("controller_manager/load_controller"),
      switch_service_name("controller_manager/switch_controller"),
      controller_name(controller_name_),
      init_done(false),
      load_client(
          nh->serviceClient<cm_msgs::LoadController>(load_service_name)),
      switch_client(
          nh->serviceClient<cm_msgs::SwitchController>(switch_service_name)),
      params_pub(nh->advertise<softgrasp_ros_controllers::DownwardForceParams>(
          controller_name_+"/downward_force_params", 1)) {
  if (ros::param::get("~force_control/desired_mass", desired_mass)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME, "desired_mass = " << desired_mass);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "~force_control/desired_mass not set");
    ros::shutdown();
  }
  if (ros::param::get("~force_control/kp", kp)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME, "kp = " << kp);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "~force_control/kp not set");
    ros::shutdown();
  }
  if (ros::param::get("~force_control/ki", ki)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME, "ki = " << ki);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "~force_control/ki not set");
    ros::shutdown();
  }
  if (ros::param::get("~force_control/kd", kd)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME, "kd = " << kd);
  } else {
    ROS_ERROR_NAMED(ROS_NAME, "~force_control/kd not set");
    ros::shutdown();
  }
}

bool ArmForceController::init() {
  ROS_INFO_NAMED(ROS_NAME, "Waiting for load controller service...");
  load_client.waitForExistence();
  ROS_INFO_NAMED(ROS_NAME, "Waiting for switch controller service...");
  switch_client.waitForExistence();

  // load the controller but don't start it
  cm_msgs::LoadController srv;
  srv.request.name = controller_name;
  if (!load_client.call(srv)) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
                           "Could not call service " << load_service_name);
    return init_done;
  }
  if (!srv.response.ok) {
    ROS_ERROR_STREAM_NAMED(
        ROS_NAME,
        "controller_manager could not load controller " << controller_name);
    return init_done;
  }

  // give some time for the controller's callback spinner to start
  ros::Duration(0.5).sleep();

  // relax the collision thresholds
  franka_msgs::SetForceTorqueCollisionBehavior csrv;
  csrv.request.lower_torque_thresholds_nominal = {
      {100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}};
  csrv.request.upper_torque_thresholds_nominal = {
      {100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}};
  csrv.request.lower_force_thresholds_nominal = {
      {100.0, 100.0, 100.0, 100.0, 100.0, 100.0}};
  csrv.request.upper_force_thresholds_nominal = {
      {100.0, 100.0, 100.0, 100.0, 100.0, 100.0}};
  auto collision_client =
      nh->serviceClient<franka_msgs::SetForceTorqueCollisionBehavior>(
          "franka_control/set_force_torque_collision_behavior");
  if (!collision_client.call(csrv)) {
    ROS_ERROR_NAMED(
        ROS_NAME,
        "Could not call franka_control/set_force_torque_collision_behavior");
    return init_done;
  }
  if (!csrv.response.success) {
    ROS_ERROR_STREAM_NAMED(
        ROS_NAME,
        "call to franka_control/set_force_torque_collision_behavior failed: "
            << csrv.response.error);
    return init_done;
  }
  

  // publish the downward force the controller should exert when it is started
  softgrasp_ros_controllers::DownwardForceParams params_msg;
  params_msg.desired_mass = desired_mass;
  params_msg.kp = kp;
  params_msg.ki = ki;
  params_msg.kd = kd;
  params_pub.publish(params_msg);
  
  init_done = true;
  return init_done;
}

bool ArmForceController::start() {
  ROS_INFO_NAMED(ROS_NAME, "Starting force controller...");
  cm_msgs::SwitchController srv;
  srv.request.start_controllers.push_back(controller_name);
  srv.request.start_asap = 0;
  srv.request.strictness = 2;
  srv.request.timeout = 7.0;
  if (!switch_client.call(srv)) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
                           "Could not call service " << switch_service_name);
    return false;
  }
  if (srv.response.ok) {
    ROS_INFO_NAMED(ROS_NAME, "Done");
  } else {
    ROS_ERROR_STREAM_NAMED(
        ROS_NAME,
        "controller_manager could not start controller " << controller_name);
    return false;
  }
  return true;
}

bool ArmForceController::stop() {
  // zero downward force
  ROS_INFO_NAMED(ROS_NAME, "Zeroing out force controller...");
  softgrasp_ros_controllers::DownwardForceParams params_msg;
  params_msg.desired_mass = 0.f;
  params_msg.kp = kp;
  params_msg.ki = ki;
  params_msg.kd = kd;
  params_pub.publish(params_msg);
  ros::Duration(5.0).sleep();
  ROS_INFO_STREAM_NAMED(ROS_NAME, "Done");

  ROS_INFO_STREAM_NAMED(ROS_NAME, "Stopping force controller...");
  cm_msgs::SwitchController srv;
  srv.request.stop_controllers.push_back(controller_name);
  srv.request.start_asap = 0;
  srv.request.strictness = 2;
  srv.request.timeout = 0.0;
  if (!switch_client.call(srv)) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
                           "Could not call service " << switch_service_name);
    return false;
  }
  if (srv.response.ok) {
    ROS_INFO_NAMED(ROS_NAME, "Done");
  } else {
    ROS_ERROR_STREAM_NAMED(
        ROS_NAME,
        "controller_manager could not stop controller " << controller_name);
    return false;
  }
  return true;
}