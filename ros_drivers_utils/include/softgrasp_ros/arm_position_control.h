#pragma once
#include <eigen3/Eigen/Core>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <string>

class ArmPositionController {
 private:
  std::string ROS_NAME;
  std::string planning_group_name;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface>
      planning_scene_interface;
  std::shared_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools;
  const moveit::core::JointModelGroup* joint_model_group;
  bool init_done;
  bool plan_display_go();
 public:
  ArmPositionController();
  bool init();
  bool home();
  bool eef_pose(const Eigen::Affine3d &bTf);
  Eigen::Affine3d get_eef();
};