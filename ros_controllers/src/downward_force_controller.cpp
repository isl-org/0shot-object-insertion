#include "softgrasp_ros_controllers/downward_force_controller.h"

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

namespace softgrasp_ros_controllers {
bool DownwardForceController::init(
    hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
  std::vector<std::string> joint_names;
  std::string arm_id;
  ROS_WARN_NAMED(ROS_NAME, 
      "Make sure your robot's endeffector is in contact "
      "with a horizontal surface before starting the controller!");
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_NAMED(ROS_NAME, "Could not read parameter arm_id");
    return false;
  }
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR_NAMED(ROS_NAME,
        "Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME, "Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
        "Exception getting model handle from interface: " << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME, "Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
        "Exception getting state handle from interface: " << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME, "Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM_NAMED(ROS_NAME, "Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  params_sub = node_handle.subscribe("downward_force_params", 1,
                                     &DownwardForceController::params_cb, this);

  return true;
}

void DownwardForceController::starting(const ros::Time& /*time*/) {
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
  // Bias correction for the current external torque
  tau_ext_initial_ = tau_measured - gravity;
  tau_error_.setZero();
  tau_error_prev_.setZero();
}

void DownwardForceController::update(const ros::Time& /*time*/, const ros::Duration& period) {
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());

  Eigen::VectorXd tau_d(7), desired_force_torque(6), tau_cmd(7), tau_ext(7),
      this_tau_error(7);
  desired_force_torque.setZero();
  desired_force_torque(2) = desired_mass_ * -9.81;
  tau_ext = tau_measured - gravity - tau_ext_initial_;
  tau_d << jacobian.transpose() * desired_force_torque;
  this_tau_error << (tau_d - tau_ext);
  tau_error_ = tau_error_ + period.toSec() * this_tau_error;
  // FF + PI control (PI gains are initially all 0)
  tau_cmd = tau_d
          + k_p_ * this_tau_error
          + k_i_ * tau_error_;
          // + k_d_ * (this_tau_error - tau_error_prev_) / period.toSec();
  // tau_error_prev_ << this_tau_error;
  tau_cmd << saturateTorqueRate(tau_cmd, tau_J_d);

  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_cmd(i));
  }

  // Update signals changed online through dynamic reconfigure
  desired_mass_ = filter_gain_ * target_mass_ + (1 - filter_gain_) * desired_mass_;
  k_p_ = filter_gain_ * target_k_p_ + (1 - filter_gain_) * k_p_;
  k_i_ = filter_gain_ * target_k_i_ + (1 - filter_gain_) * k_i_;
  k_d_ = filter_gain_ * target_k_d_ + (1 - filter_gain_) * k_d_;
}

void DownwardForceController::params_cb(const DownwardForceParams::ConstPtr& msg) {
  target_mass_ = static_cast<double>(msg->desired_mass);
  target_k_p_ = static_cast<double>(msg->kp);
  target_k_i_ = static_cast<double>(msg->ki);
  target_k_d_ = static_cast<double>(msg->kd);
  ROS_INFO_STREAM_NAMED(ROS_NAME, "callback: m = " << target_mass_
                                                   << ", kp = " << target_k_p_
                                                   << ", ki = " << target_k_i_
                                                   << ", kd = " << target_k_d_);
}

Eigen::Matrix<double, 7, 1> DownwardForceController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

}  // namespace softgrasp_ros_controllers

PLUGINLIB_EXPORT_CLASS(softgrasp_ros_controllers::DownwardForceController,
                       controller_interface::ControllerBase)