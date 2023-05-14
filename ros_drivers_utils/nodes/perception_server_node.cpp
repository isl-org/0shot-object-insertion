#include <ros/ros.h>
#include "softgrasp_ros/perception_server.h"


int main(int argc, char **argv) {
  ros::init(argc, argv, "perception_server");
  ros::NodeHandlePtr nh = boost::make_shared<ros::NodeHandle>();
  bool debug_mode;
  ros::param::param<bool>("~debug_mode", debug_mode, false);

  PerceptionServer server(nh, debug_mode);
  ros::spin();
  return 0;
}