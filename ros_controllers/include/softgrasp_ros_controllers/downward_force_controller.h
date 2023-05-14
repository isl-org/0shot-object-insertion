#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/ros.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include "softgrasp_ros_controllers/DownwardForceParams.h"
#include <Eigen/Core>


namespace softgrasp_ros_controllers {
class DownwardForceController
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaModelInterface,
          hardware_interface::EffortJointInterface,
          franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>&
          tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  double desired_mass_{0.5};
  double target_mass_{0.0};
  double k_p_{0.0};
  double k_i_{0.0};
  double k_d_{0.0};
  double target_k_p_{1.0};
  double target_k_i_{2.0};
  double target_k_d_{0.0};
  double filter_gain_{0.001};
  Eigen::Matrix<double, 7, 1> tau_ext_initial_;
  Eigen::Matrix<double, 7, 1> tau_error_, tau_error_prev_;
  static constexpr double kDeltaTauMax{1.0};
  std::string ROS_NAME{"DownwardForceController"};

  ros::Subscriber params_sub;
  void params_cb(const DownwardForceParams::ConstPtr& msg);
};

}  // namespace softgrasp_ros_controllers