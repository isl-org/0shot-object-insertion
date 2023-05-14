#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include <actionlib/client/simple_action_client.h>
#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <Eigen/Dense>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

#include <softgrasp_ros_controllers/compliance_paramConfig.h>
#include <softgrasp_ros_controllers/ManageExperimentAction.h>
#include <softgrasp_ros_controllers/TFPolicyAction.h>
#include <tf2_ros/transform_listener.h>

namespace softgrasp_ros_controllers {

class TFPolicyController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  typedef actionlib::SimpleActionClient<softgrasp_ros_controllers::TFPolicyAction> PolicyActionClient;
  typedef actionlib::SimpleActionClient<softgrasp_ros_controllers::ManageExperimentAction> ManageExperimentClient;

  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1> &tau_d_calculated,
                                                 const Eigen::Matrix<double, 7, 1> &tau_J_d);

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  double filter_params_{0.005};
  double nullspace_stiffness_{0.5};
  double nullspace_stiffness_target_{0.5};
  const double delta_tau_max_{1.0};
  double K_P{1.0}, K_I{0.0};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  Eigen::Matrix<double, 7, 1> tau_error_;
  Eigen::Vector3d position_d_, position_;
  Eigen::Quaterniond orientation_d_, orientation_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;
  int decision_step_, count_since_last_policy_inference_, inference_count_{0};
  Eigen::Vector3d O_T_G_pos_{Eigen::Vector3d::Identity()};
  Eigen::Quaterniond O_T_G_quat_{Eigen::Quaterniond::Identity()};
  float inference_speed_{0.f};
  std::mutex target_mutex_;
  double circle_r_{0.05}, circle_time_{5.0};
  Eigen::Vector3d circle_c_;
  ros::Duration elapsed_time_;
  Eigen::Matrix<double, 7, 1> tau_ext_initial_;
  tf2_ros::Buffer tfBuffer;
  std::unique_ptr<tf2_ros::TransformListener> tfListener;
  bool episode_ended;
  double success_thresh_x, success_thresh_y, success_thresh_z;

  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<softgrasp_ros_controllers::compliance_paramConfig>>
      dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(softgrasp_ros_controllers::compliance_paramConfig& config, uint32_t level);

  // TF policy client
  std::unique_ptr<PolicyActionClient> action_client_;
  // release grasp client
  std::unique_ptr<ManageExperimentClient> manage_experiment_client_;
  
  // policy inference callback
  void actionCb(const actionlib::SimpleClientGoalState &state,
                const softgrasp_ros_controllers::TFPolicyResultConstPtr &result);

  // used for debugging, sets the goal to the next waypoint on a predefined debugging trajectory e.g. circle
  void updateGoal();
};

}  // namespace softgrasp_ros_controllers
