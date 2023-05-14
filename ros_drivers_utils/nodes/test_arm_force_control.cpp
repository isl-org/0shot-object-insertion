#include <ros/ros.h>
#include "softgrasp_ros/arm_force_control.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_arm_force_control");
  ros::NodeHandlePtr nh = boost::make_shared<ros::NodeHandle>();
  ros::AsyncSpinner spinner(4);
  spinner.start();

  std::string controller_name;
  if (!ros::param::get("~force_control/controller_name", controller_name)) {
    ROS_ERROR("force_control/controller_name not set");
    return -1;
  }
  ArmForceController controller(nh, controller_name);
  controller.init();
  ros::Duration(5.0).sleep();
  ROS_INFO("Starting force controller...");
  controller.start();
  ros::Duration(20.0).sleep();
  ROS_INFO("Stopping force controller...");
  controller.stop();
  ros::waitForShutdown();
  return 0;
}