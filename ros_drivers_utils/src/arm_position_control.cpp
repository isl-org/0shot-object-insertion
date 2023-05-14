#include <actionlib_msgs/GoalStatusArray.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include "softgrasp_ros/arm_position_control.h"
#include <tf2_eigen/tf2_eigen.h>

using namespace ros;
namespace mpi = moveit::planning_interface;

ArmPositionController::ArmPositionController()
    : planning_group_name("panda_arm"),
      ROS_NAME("ArmPositionController"),
      init_done(false) {}

bool ArmPositionController::init() {
  if (!ros::topic::waitForMessage<actionlib_msgs::GoalStatusArray>(
      "move_group/status", ros::Duration(10.0))) {
    ROS_ERROR_STREAM_NAMED(ROS_NAME,
                           "Did not receive a messge on move_group/status");
    init_done = false;
  } else {
    move_group = std::make_shared<mpi::MoveGroupInterface>(planning_group_name);
    planning_scene_interface = std::make_shared<mpi::PlanningSceneInterface>();
    visual_tools =
        std::make_shared<moveit_visual_tools::MoveItVisualTools>("panda_link0");
    joint_model_group =
        move_group->getCurrentState()->getJointModelGroup(planning_group_name);

    visual_tools->deleteAllMarkers();
    visual_tools->loadRemoteControl();
    visual_tools->trigger();
    init_done = true;
  }
  return init_done;
}

bool ArmPositionController::home() {
  if (!init_done) {
    ROS_WARN_NAMED(ROS_NAME, "Not initialized");
    return false;
  }
  move_group->setNamedTarget("ready");
  return plan_display_go();
}

bool ArmPositionController::eef_pose(const Eigen::Affine3d &bTf) {
  if (!init_done) {
    ROS_WARN_NAMED(ROS_NAME, "Not initialized");
    return false;
  }
  Eigen::Isometry3d bTf_i;
  bTf_i.translation() = bTf.translation();
  bTf_i.linear() = bTf.rotation();
  move_group->setPoseTarget(bTf_i);
  return plan_display_go();
}

Eigen::Affine3d ArmPositionController::get_eef() {
  geometry_msgs::PoseStamped bTf = move_group->getCurrentPose();
  Eigen::Affine3d bTf_out;
  tf2::fromMsg(bTf.pose, bTf_out);
  return bTf_out;
}

bool ArmPositionController::plan_display_go() {
  mpi::MoveGroupInterface::Plan plan;
  bool success = move_group->plan(plan) == mpi::MoveItErrorCode::SUCCESS;
  ROS_INFO_NAMED(ROS_NAME, "Visualizing plan %s", success ? "" : "FAILED");
  auto target = move_group->getPoseTarget();
  visual_tools->publishAxisLabeled(target.pose, "pose goal");
  visual_tools->publishTrajectoryLine(plan.trajectory_, joint_model_group);
  visual_tools->trigger();
  visual_tools->prompt(
      "Press 'next' in the RvizVisualToolsGui window to continue");
  success = move_group->move() == mpi::MoveItErrorCode::SUCCESS;
  move_group->clearPoseTargets();
  return success;
}