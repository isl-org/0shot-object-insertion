#include <ros/ros.h>
#include "softgrasp_ros/hand_control.h"


int main(int argc, char **argv) {
  ros::init(argc, argv, "test_hand_control");
  ros::NodeHandlePtr nh = boost::make_shared<ros::NodeHandle>();
  ros::AsyncSpinner spinner(4);
  spinner.start();

  HandController controller(nh);
  controller.init();
  float opening_amount(3e-2f), grip_strength;
  ros::param::param<float>("~grip_strength", grip_strength, 0.5f);
  ROS_INFO("Opening amount = %f", opening_amount);
  controller.set_params(0.5f, opening_amount);
  controller.set_state(true);
  opening_amount = 8e-2f;
  ROS_INFO("Opening amount = %f", opening_amount);
  controller.set_params(0.5f, opening_amount);
  controller.set_state(true);
  controller.set_state(false);
  return 0;
}