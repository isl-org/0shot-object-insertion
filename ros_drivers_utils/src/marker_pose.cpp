#include "softgrasp_ros/marker_pose.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d.hpp>
#include <tf2/LinearMath/Matrix3x3.h>


void drawAxis(cv::InputOutputArray _image, cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
              cv::InputArray _rvec, cv::InputArray _tvec, float length) {
  CV_Assert(_image.getMat().total() != 0 && (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
  CV_Assert(length > 0);

  // project axis points
  std::vector<cv::Point3f> axis_points;
  axis_points.push_back(cv::Point3f(0, 0, 0));
  axis_points.push_back(cv::Point3f(length, 0, 0));
  axis_points.push_back(cv::Point3f(0, length, 0));
  axis_points.push_back(cv::Point3f(0, 0, length));
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(axis_points, _rvec, _tvec, _cameraMatrix, _distCoeffs, image_points);

  // draw axis lines
  cv::line(_image, image_points[0], image_points[1], cv::Scalar(255, 0, 0), 3);
  cv::line(_image, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 3);
  cv::line(_image, image_points[0], image_points[3], cv::Scalar(0, 0, 255), 3);
}


MarkerPose::MarkerPose(const ros::NodeHandle &nh) :
  nh_(nh),
  it_(std::make_unique<image_transport::ImageTransport>(nh_)),
  sub_(it_->subscribeCamera("image", 1, &MarkerPose::camera_cb, this)),
  pub_(it_->advertise("image_marker_pose", 1)),
  parameters(cv::aruco::DetectorParameters::create()),
  dictionary(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250)),
  board(cv::aruco::GridBoard::create(3, 2, marker_size, 0.0075, dictionary)) {
    parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
}


void MarkerPose::camera_cb(const sensor_msgs::ImageConstPtr &im_msg, const sensor_msgs::CameraInfoConstPtr &cinfo) {
  if (K_.empty() || D_.empty()) {
    K_ = cv::Mat(3, 3, CV_64F);
    D_ = cv::Mat::zeros(5, 1, CV_64F);

    for (int i=0; i<9; i++) {
      K_.at<double>(i/3, i%3) = cinfo->K[i];
    }

    if (cinfo->distortion_model != std::string("plumb_bob")) {
      ROS_ERROR_STREAM("Distortion model " << cinfo->distortion_model << " is invalid");
      return;
    } else {
      for (int i=0; i<5; i++) D_.at<double>(i, 0) = cinfo->D[i];
    }
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(im_msg);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge_exception: %s", e.what());
    return;
  }
  cv::Mat im = cv_ptr->image;
  cv::Mat im_show = im;

  cv::aruco::detectMarkers(im, dictionary, markerCorners, markerIds, parameters, rejectedCandidates, K_, D_);
  if (markerIds.empty()) {
    ROS_WARN_THROTTLE(1.0, "No marker detected");
  } else {
    if (markerIds.size() == 1) {
      std::vector<cv::Vec3d> rvecs, tvecs;
      cv::aruco::estimatePoseSingleMarkers(markerCorners, marker_size, K_, D_, rvecs, tvecs);
      tvec_ = tvecs[0];
      rvec_ = rvecs[0];
      // 180 degrees around X axis, followed by -90 around Z axis
      cv::Mat offset(cv::Mat::eye(3, 3, CV_64FC1)), R(3, 3, CV_64FC1);
      offset.at<double>(0, 0) = 0.0;
      offset.at<double>(1, 1) = 0.0;
      offset.at<double>(0, 1) = 1.0;
      offset.at<double>(1, 0) = 1.0;
      offset.at<double>(2, 2) = -1.0;
      cv::Rodrigues(rvec_, R);
      R = R * offset;
      cv::Rodrigues(R, rvec_);
    } else {
      cv::aruco::refineDetectedMarkers(im, board, markerCorners, markerIds, rejectedCandidates, K_, D_);
      int n_used = cv::aruco::estimatePoseBoard(markerCorners, markerIds, board, K_, D_, rvec_, tvec_, false);
      if (!n_used) {
        ROS_WARN_STREAM_THROTTLE(1.0, "" << markerIds.size() << " markers detected, but board pose estimation failure");
        return;
      }
    }
    cv::aruco::drawDetectedMarkers(im_show, markerCorners);
    drawAxis(im_show, K_, D_, rvec_, tvec_, 0.05);
  }
  
  pub_.publish(cv_bridge::CvImage(im_msg->header, im_msg->encoding, im_show).toImageMsg());

  cv::Mat R(3, 3, CV_64FC1);
  cv::Rodrigues(rvec_, R);
  tf2::Matrix3x3 tf2_R(
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2)
  );
  tf2::Quaternion q;
  tf2_R.getRotation(q);
  ROS_INFO_STREAM_THROTTLE(0.5, "Pose = " << tvec_[0] << " " << tvec_[1] << " " << tvec_[2] << "  "
                                          << q.getX() << " " << q.getY() << " " << q.getZ() << " " << q.getW());
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "marker_pose");
  ros::NodeHandle nh;

  MarkerPose mp(nh);
  ros::spin();
  return 0;
}