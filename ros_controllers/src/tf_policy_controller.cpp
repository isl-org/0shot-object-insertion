#include <softgrasp_ros_controllers/tf_policy_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <geometry_msgs/TransformStamped.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka_example_controllers/pseudo_inversion.h>

#include <softgrasp_ros_controllers/TFPolicyGoal.h>
#include <sstream>

namespace softgrasp_ros_controllers {

bool TFPolicyController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("TFPolicyController: Could not read parameter arm_id");
    return false;
  }
  std::string policy_action_name;
  if (!node_handle.getParam("policy_action_name", policy_action_name)) {
    ROS_ERROR_STREAM("TFPolicyController: Could not read parameter policy_action_name");
    return false;
  }
  action_client_ = std::make_unique<PolicyActionClient>(node_handle, policy_action_name, false);
  
  manage_experiment_client_ = std::make_unique<ManageExperimentClient>("/manage_experiment", false);

  if (!node_handle.getParam("decision_step", decision_step_)) {
    ROS_ERROR_STREAM("TFPolicyController: Could not read parameter decision_step");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "TFPolicyController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  if (
    !node_handle.getParam("success_thresh_x", success_thresh_x) ||
    !node_handle.getParam("success_thresh_y", success_thresh_y) ||
    !node_handle.getParam("success_thresh_z", success_thresh_z)
  ) {
    ROS_ERROR("TFPolicyController: could not get success_thresh_{x,y,z} params");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "TFPolicyController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "TFPolicyController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "TFPolicyController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "TFPolicyController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "TFPolicyController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "TFPolicyController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "dynamic_reconfigure_compliance_param_node");
  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<softgrasp_ros_controllers::compliance_paramConfig>>(
      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&TFPolicyController::complianceParamCallback, this, _1, _2));

  position_.setZero();
  orientation_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_ = 1200.0 * Eigen::Matrix<double, 6, 6>::Identity();
  cartesian_damping_ = 4.0 * sqrt(1200.0) * Eigen::Matrix<double, 6, 6>::Identity();

  tau_error_.setZero();

  // get goal pose
  tfListener = std::make_unique<tf2_ros::TransformListener>(tfBuffer);
  geometry_msgs::TransformStamped O_T_G_tf;
  try {
    O_T_G_tf = tfBuffer.lookupTransform("panda_link0", "goal", ros::Time::now(), ros::Duration(5.0));
  } catch (tf2::TransformException &e) {
    ROS_ERROR_STREAM("TFPolicyController: " << e.what());
    return false;
  }
  const auto &q = O_T_G_tf.transform.rotation;
  O_T_G_quat_ = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
  const auto &t = O_T_G_tf.transform.translation;
  O_T_G_pos_ << t.x, t.y, t.z; // - 0.08;  // uncomment to match the ebar_T_e(z) at bottom touch
  
  return true;
}

void TFPolicyController::starting(const ros::Time& /*time*/) {
  // initialize various quantities
  franka::RobotState initial_state = state_handle_->getRobotState();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_measured(initial_state.tau_J.data());
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
  tau_ext_initial_ = tau_measured - gravity;

  // set equilibrium point to current state
  position_ = initial_transform.translation();
  orientation_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());

  // comment the following two lines for real mode
  // O_T_G_pos_ = position_;
  // O_T_G_quat_ = orientation_;

  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ = q_initial;
  
  ROS_INFO_STREAM("TFPolicyController: waiting for TFPolicyActionServer...");
  action_client_->waitForServer();
  ROS_INFO_STREAM("TFPolicyController: found TFPolicyActionServer");
  ROS_INFO_STREAM("TFPolicyController: waiting for ManageExperimentActionServer...");
  manage_experiment_client_->waitForServer();
  ROS_INFO_STREAM("TFPolicyController: found ManageExperimentActionServer");
  count_since_last_policy_inference_ = 0;
  circle_c_ = position_d_;
  circle_c_.x() -= circle_r_;
  elapsed_time_ = ros::Duration(0.0);
  episode_ended = false;
}

void TFPolicyController::update(const ros::Time& time, const ros::Duration& period) {
  elapsed_time_ += period;

  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 49> mass_array = model_handle_->getMass();

  // convert to Eigen
  const Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
  const Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  const Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  const Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  const Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  const Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J(robot_state.tau_J.data());
  const Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());
  const Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_ext_filtered(robot_state.tau_ext_hat_filtered.data());
  const Eigen::Map<Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
  const Eigen::Affine3d O_T_EE(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  position_ = O_T_EE.translation();
  orientation_ = Eigen::Quaterniond(O_T_EE.linear());

  Eigen::Matrix<double, 6, 1> error;
  if (!episode_ended) {
    // send observation to policy
    if (count_since_last_policy_inference_ % decision_step_ == 0) {
      Eigen::Vector3d pos(O_T_G_pos_ - position_);
      // ROS_INFO_STREAM("Translation error = " << t.norm());

      // detect if episode should be ended
      // ROS_INFO_STREAM_THROTTLE(0.1, "" << pos.x() << " " << pos.y() << " " << pos.z());
      if (
        (std::abs(pos.x()) <= success_thresh_x) &&
        (std::abs(pos.y()) <= success_thresh_y) &&
        (pos.z() >= -success_thresh_z)
      ) {
        episode_ended = true;
        ROS_INFO_THROTTLE(0.5, "episode ended");
        softgrasp_ros_controllers::ManageExperimentGoal goal;
        goal.stage_id = 2;  // release grasp and stop policy controller
        manage_experiment_client_->sendGoal(goal);
      }

      if (!episode_ended) {
        Eigen::Quaterniond quat(O_T_G_quat_ * orientation_.inverse());
        if (quat.w() < 0.0) quat.coeffs() << -quat.coeffs();  // prevent flipping
        softgrasp_ros_controllers::TFPolicyGoal goal;
        auto& o = goal.observation;
        o.header.seq = count_since_last_policy_inference_;
        o.header.stamp.sec = elapsed_time_.sec;
        o.header.stamp.nsec = elapsed_time_.nsec;
        o.g_T_ee.position.x = pos.x();
        o.g_T_ee.position.y = pos.y();
        o.g_T_ee.position.z = pos.z();
        o.g_T_ee.orientation.w = quat.w();
        o.g_T_ee.orientation.x = quat.x();
        o.g_T_ee.orientation.y = quat.y();
        o.g_T_ee.orientation.z = quat.z();
        const std::array<double, 6> F(robot_state.O_F_ext_hat_K);
        o.wrench.force.x = F[0];
        o.wrench.force.y = F[1];
        o.wrench.force.z = F[2];
        o.wrench.torque.x = F[3];
        o.wrench.torque.y = F[4];
        o.wrench.torque.z = F[5];
        action_client_->sendGoal(goal, boost::bind(&TFPolicyController::actionCb, this, _1, _2),
                                 PolicyActionClient::SimpleActiveCallback(),
                                 PolicyActionClient::SimpleFeedbackCallback());
        // updateGoal();
        count_since_last_policy_inference_ = 0;
      }
    }
    count_since_last_policy_inference_++;

    // compute error to desired pose
    // position error
    error.head(3) << position_ - position_d_;

    // orientation error
    if (orientation_d_.coeffs().dot(orientation_.coeffs()) < 0.0) orientation_.coeffs() << -orientation_.coeffs();
    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation_.inverse() * orientation_d_);
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
    // Transform to base frame
    error.tail(3) << -O_T_EE.linear() * error.tail(3);
  } else {
    error << 0.0, 0.0, 0.01, 0.0, 0.0, 0.0;
  }
  
  Eigen::Matrix<double, 6, 1> vel(jacobian * dq);
  Eigen::Vector3d desired_force = -cartesian_stiffness_.topLeftCorner(3, 3) * error.head(3) -
      cartesian_damping_.topLeftCorner(3, 3) * vel.head(3);
  // Eigen::Vector3d ori_error(orientation_error(orientation_d_, orientation_));
  Eigen::Vector3d desired_torque = -cartesian_stiffness_.bottomRightCorner(3, 3) * error.tail(3) -
      cartesian_damping_.bottomRightCorner(3, 3) * vel.tail(3);

  // matrices used in operational space control
  Eigen::Matrix<double, 7, 7> mass_inv(mass.inverse());
  Eigen::MatrixXd lambda_full_inv(jacobian * mass_inv * jacobian.transpose()), lambda_full;
  franka_example_controllers::pseudoInverse(lambda_full_inv, lambda_full);
  Eigen::MatrixXd lambda_pos_inv(jacobian.topRows(3) * mass_inv * jacobian.topRows(3).transpose()), lambda_pos;
  franka_example_controllers::pseudoInverse(lambda_pos_inv, lambda_pos);
  Eigen::MatrixXd lambda_rot_inv(jacobian.bottomRows(3) * mass_inv * jacobian.bottomRows(3).transpose()), lambda_rot;
  franka_example_controllers::pseudoInverse(lambda_rot_inv, lambda_rot);
  Eigen::MatrixXd Jbar(mass_inv * jacobian.transpose() * lambda_full);
  Eigen::MatrixXd nullspace_matrix(Eigen::MatrixXd::Identity(7, 7) - Jbar * jacobian);

  Eigen::VectorXd decoupled_wrench(6), tau_task(7), tau_nullspace(7);
  decoupled_wrench.head(3) << lambda_pos * desired_force;
  decoupled_wrench.tail(3) << lambda_rot * desired_torque;
  tau_task << jacobian.transpose() * decoupled_wrench;

  Eigen::VectorXd pose_torques(mass*nullspace_stiffness_*(q_d_nullspace_ - q) - 2.0*sqrt(nullspace_stiffness_)*dq);
  tau_nullspace << nullspace_matrix.transpose() * pose_torques;
  Eigen::VectorXd tau_d(7), tau_ext(7), tau_cmd(7);

  /*
  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7), tau_ext(7), tau_cmd(7);

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  Eigen::MatrixXd jacobian_transpose_pinv;
  franka_example_controllers::pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // Cartesian PD control with damping ratio = 1
  tau_task << jacobian.transpose() * (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));
  // nullspace PD control with damping ratio = 1
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) *
                       (nullspace_stiffness_ * (q_d_nullspace_ - q) - (2.0 * sqrt(nullspace_stiffness_)) * dq);
  */
  tau_d = tau_task + tau_nullspace; // desired torque
  // compensate external wrench (NOTE: does not work in sim!)
  tau_d << tau_d - tau_ext_filtered;
  // tau_ext = tau_J - gravity - coriolis - tau_ext_initial_;  // actual exterted torque
  // tau_error_ = tau_error_ + period.toSec() * (tau_d - tau_ext);  // integral term for PI control

  // Desired torque
  // tau_cmd << tau_d + K_P*(tau_d - tau_ext) + K_I*tau_error_;
  tau_cmd << tau_d + coriolis;
  
  // Saturate torque rate to avoid discontinuities
  tau_cmd << saturateTorqueRate(tau_cmd, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_cmd(i));
  }

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  std::lock_guard<std::mutex> target_mutex_lock(target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

Eigen::Matrix<double, 7, 1> TFPolicyController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void TFPolicyController::complianceParamCallback(
    softgrasp_ros_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 4.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 4.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
  nullspace_stiffness_target_ = config.nullspace_stiffness;
}

void TFPolicyController::actionCb(const actionlib::SimpleClientGoalState &state,
                                  const softgrasp_ros_controllers::TFPolicyResultConstPtr &result)
{
  if (state != actionlib::SimpleClientGoalState::SUCCEEDED) {
    ROS_ERROR("TFPolicyActionServer did not succeed");
    return;
  } else {
    inference_speed_ = (inference_speed_ * inference_count_ + count_since_last_policy_inference_) /
                       (inference_count_ + 1.f);
    inference_count_++;
    ROS_INFO_STREAM_THROTTLE(5.0, "Recieved policy inference in " << inference_speed_ << " timesteps");
  }

  /*
  // calculate robot motion errors
  double pos_err((position_ - position_d_target_).norm());
  Eigen::AngleAxisd ori_err;
  ori_err = orientation_.inverse() * orientation_d_target_;
  double ori_err_deg(180.0 * ori_err.angle() / M_PI);
  ROS_INFO_STREAM_THROTTLE(0.5, "Position error = " << pos_err << " m, orientation error = " << ori_err_deg);
  */

  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  std::lock_guard<std::mutex> target_mutex_lock(target_mutex_);
  
  auto action = result->action.action;
  position_d_target_ = position_ + Eigen::Vector3d(action[0], action[1], action[2]);
  orientation_d_target_ = Eigen::Quaterniond(action[6], action[3], action[4], action[5]) * orientation_;
  
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}

void TFPolicyController::updateGoal() {
  double time(elapsed_time_.toSec());
  double angle = M_PI * (1.0 - std::cos(2 * M_PI * time / circle_time_));
  std::lock_guard<std::mutex> target_mutex_lock(target_mutex_);
  position_d_target_ = circle_c_ + Eigen::Vector3d(circle_r_*std::cos(angle), circle_r_*std::sin(angle), 0.0);
}

}  // namespace softgrasp_ros_controllers

PLUGINLIB_EXPORT_CLASS(softgrasp_ros_controllers::TFPolicyController, controller_interface::ControllerBase)
