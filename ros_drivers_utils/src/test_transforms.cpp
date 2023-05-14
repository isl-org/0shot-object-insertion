#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Vector3Stamped.h>


class FrameDrawer
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_;
  image_transport::Publisher pub_;
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  image_geometry::PinholeCameraModel cam_model_;
  std::vector<std::string> frame_ids_{"panda_link8", "panda_K"};
  std::array<geometry_msgs::PointStamped, 4> frame_points_, camera_points_;

public:
  FrameDrawer(const ros::NodeHandle &nh) : nh_(nh), it_(nh_), tfListener(tfBuffer) {
    sub_ = it_.subscribeCamera("image", 1, &FrameDrawer::imageCb, this);
    pub_ = it_.advertise("image_marker_pose", 1);
    double axis_size(0.05); 
    frame_points_[0].point.x = 0.0;
    frame_points_[0].point.y = 0.0;
    frame_points_[0].point.z = 0.0;
    frame_points_[1].point.x = axis_size;
    frame_points_[1].point.y = 0.0;
    frame_points_[1].point.z = 0.0;
    frame_points_[2].point.x = 0.0;
    frame_points_[2].point.y = axis_size;
    frame_points_[2].point.z = 0.0;
    frame_points_[3].point.x = 0.0;
    frame_points_[3].point.y = 0.0;
    frame_points_[3].point.z = axis_size;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg) {
    cv_bridge::CvImageConstPtr input_bridge;
    try {
      input_bridge = cv_bridge::toCvShare(image_msg);
    }
    catch (cv_bridge::Exception& ex){
      ROS_ERROR("[draw_frames] Failed to convert image");
      return;
    }
    cv::Mat image = input_bridge->image;
    
    cam_model_.fromCameraInfo(info_msg);

    for(const std::string &frame_id: frame_ids_) {
      for (auto &p: frame_points_) {
        p.header.stamp = info_msg->header.stamp;
        p.header.frame_id = frame_id;
      } 
      try {
          for (int i=0; i<4; i++) tfBuffer.transform(frame_points_[i], camera_points_[i], cam_model_.tfFrame(), ros::Duration(0.33));
      }
      catch (tf2::TransformException& ex) {
        ROS_WARN("[test_transforms] TF exception:\n%s", ex.what());
        return;
      }

      cv::Point3d c_o(camera_points_[0].point.x, camera_points_[0].point.y, camera_points_[0].point.z);
      cv::Point2d c_p = cam_model_.project3dToPixel(c_o);
      for (int i=0; i<3; i++) {
        cv::Point3d c_o1(camera_points_[i+1].point.x, camera_points_[i+1].point.y, camera_points_[i+1].point.z);
        cv::Point2d c_p1 = cam_model_.project3dToPixel(c_o1);
        cv::Scalar color(0.0, 0.0, 0.0, 255.0);
        color[i] = 255.0;
        cv::line(image, c_p, c_p1, color, 3, cv::LINE_AA);
      }

      static const int RADIUS = 3;
      int baseline;
      cv::Size text_size = cv::getTextSize(frame_id.c_str(), cv::FONT_HERSHEY_SIMPLEX, 1.0, 1.0, &baseline);
      cv::Point origin(c_p.x - text_size.width/2, c_p.y - RADIUS - baseline - 3);
      cv:putText(image, frame_id.c_str(), origin, cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255,0,0));
    }

    pub_.publish(input_bridge->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_transforms");
  ros::NodeHandle nh;
  FrameDrawer drawer(nh);
  ros::spin();
}