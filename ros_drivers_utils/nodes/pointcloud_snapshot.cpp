// saves a point cloud snapshot

#include <open3d/Open3D.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <boost/filesystem.hpp>
#include <sstream>

#include "softgrasp_ros/geometry_utils.h"


using namespace std;
using namespace ros;
namespace fs = boost::filesystem;

class PointCloudSnapshotter {
 private:
  NodeHandlePtr nh;
  Subscriber cloud_sub;
  void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &msg);
  ServiceServer save_service;
  bool save_cb(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);

  string prefix;
  string ROS_NAME;

  bool save;
  size_t count;

 public:
  PointCloudSnapshotter(const NodeHandlePtr &nh_);
};

PointCloudSnapshotter::PointCloudSnapshotter(const NodeHandlePtr &nh_)
    : nh(nh_),
      ROS_NAME("PointCloudSnapshotter"),
      save(false),
      count(0),
      cloud_sub(
          nh->subscribe("points", 1, &PointCloudSnapshotter::cloud_cb, this)),
      save_service(nh->advertiseService(
          "save_pointcloud", &PointCloudSnapshotter::save_cb, this)) {
  nh->param<string>("prefix", prefix, ".");
  ROS_INFO_STREAM_NAMED(ROS_NAME, "Prefix = " << prefix);
}

void PointCloudSnapshotter::cloud_cb(
    const sensor_msgs::PointCloud2ConstPtr &msg) {
  if (!save) return;
  auto cloud = pc_ros2open3d(*msg);
  stringstream ss;
  ss << "cloud" << setw(3) << setfill('0') << count << ".pcd";
  fs::path filename(prefix);
  filename /= ss.str();
  if (open3d::io::WritePointCloud(filename.string(), cloud)) {
    ROS_INFO_STREAM_NAMED(ROS_NAME, "Saved " << filename);
    save = false;
  } else {
    ROS_ERROR_STREAM_NAMED(ROS_NAME, "Could not save " << filename);
  }
}

bool PointCloudSnapshotter::save_cb(std_srvs::Empty::Request &req,
                                    std_srvs::Empty::Response &resp) {
  ROS_INFO_NAMED(ROS_NAME, "Save service called");
  save = true;
  return true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "point_cloud_snapshotter");
  NodeHandlePtr nh = boost::make_shared<NodeHandle>();

  PointCloudSnapshotter ps(nh);
  ros::spin();
  return 0;
}