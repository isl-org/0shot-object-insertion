#pragma once
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>


class MarkerPose {
  double marker_size{0.075};
  ros::NodeHandle nh_;
  std::unique_ptr<image_transport::ImageTransport> it_;
  image_transport::CameraSubscriber sub_;
  image_transport::Publisher pub_;
  cv::Vec3d rvec_, tvec_;
  cv::Mat K_, D_;

  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters;
  cv::Ptr<cv::aruco::Dictionary> dictionary;
  cv::Ptr<cv::aruco::Board> board;
  
  void camera_cb(const sensor_msgs::ImageConstPtr &im, const sensor_msgs::CameraInfoConstPtr &cinfo);
 
 public:
  MarkerPose(const ros::NodeHandle &nh);
};