#include <ros/ros.h>
#include "softgrasp_ros/arm_position_control.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_arm_position_control");
  ros::NodeHandlePtr nh = boost::make_shared<ros::NodeHandle>();
  ros::AsyncSpinner spinner(4);
  spinner.start();

  ArmPositionController controller;
  controller.init();
  controller.home();
  Eigen::Affine3d bTf = controller.get_eef();
  bTf.translation().x() += 0.1f;
  controller.eef_pose(bTf);
  controller.home();
  return 0;
}