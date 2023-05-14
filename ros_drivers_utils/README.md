# Installation
Create a ROS [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) with the following packages:
  - this (`softgrasp_ros`),
  - [`softgrasp_ros_controllers`](../ros_controllers),
  - our [fork of `franka_ros`](https://github.com/samarth-robo/softgrasp_franka_ros/commits/softgrasp)

# Documentation

## Gripper Operation
- Start soft gripper: `roslaunch softgrasp_ros hand_control_interface.launch`
- Grasp params: `rosservice call /grasp_params 0.5 0.5`
- Grasp close: `rosservice call /grasp_state true`, grasp open: `rosservice call /grasp_state false`

## Moving to start pose
`roslaunch softgrasp_ros move_to_start_pose.launch n:=0`, replace `0` with pose number

## Coordinate frames:
- World = robot base. Orientation: X facing front of the robot, Z up, Y left
- Stiffness frame:
  - Position at the contact point between plate and fingers
  - Orientation: X perpendicular to finger closing motion facing away from the robot, Z down, Y right
  - Sim: `env_eef_{xpos,xquat}`, Real: `RobotState.O_T_EE` (`RobotState.EE_T_K` is identity)

## Softgrasp TF Policy
- Launch TF policy on real robot:
  - Specify policy and checkpoint in `softgrasp_ros_controllers/real.launch`
  - Launch policy: `roslaunch softgrasp_ros_controllers real.launch policy_name:=softgrasp`. It starts a TF `softgrasp`
    policy action server
  - You can change `policy_name` to `zero` for straight line to goal
  - Switch from `cartesian_impedance_example_controller` to `tf_policy_controller` using
  ```
  $ rosservice call /controller_manager/switch_controller "start_controllers: ['tf_policy_controller']                                                       
  stop_controllers: ['cartesian_impedance_example_controller']
  strictness: 0
  start_asap: false
  timeout: 0.0"
  ```
- In above launch commands, replace `real.launch` with `sim.launch` to launch the same policy in Gazebo

  ## Testing the TF policy in Gazebo
  - `roslaunch softgrasp_ros_controllers sim.launch policy_name:=softgrasp`
  - Teleoperate through RViz to grasp the stone and lift it up
  - Close the grasp by `rostopic pub --once /franka_gripper/grasp/goal franka_gripper/GraspActionGoal "goal: { width: 0.03, epsilon:{ inner: 0.005, outer: 0.005 }, speed: 0.1, force: 5.0}"`
  - Open the grasp by `rostopic pub --once /franka_gripper/move/goal franka_gripper/MoveActionGoal "goal: { width: 0.08, speed: 0.1 }"`
  - Switch from `cartesian_impedance_example_controller` to `tf_policy_controller` using commands in previous section

## Experiments
- Move to a pre-sampled random start pose: `roslaunch softgrasp_ros move_to_start_pose.launch n:=6` (optional)
- Move to goal vicinity: `roslaunch softgrasp_ros run_experiment.launch policy_name:=<softgrasp|downward|random_search|zero>`.
  This also starts a TF `downward` policy action server 
- Switch from `position_joint_trajectory_controller` to `tf_policy_controller` (note, command is different from above,
  which stops the `cartesian_impedance_example_controller`):
```
$ rosservice call /controller_manager/switch_controller "start_controllers: ['tf_policy_controller']                                                       
stop_controllers: ['position_joint_trajectory_controller']
strictness: 0
start_asap: false
timeout: 0.0"
```
- In the `rqt_controller_manager` window, stop the running `position_joint_trajectory_controller` and start the
`downward_force_controller`

## Policies used
- Translational stiffness = rotational stiffness = 1200
- translational damping = rotational damping = 4 * sqrt(1200)
- translational action scaling +- 0.15
- translational action scaling +- 0.5
- force scaling = 15.0
- torque scaling = 7.5
- ours (full), ours w/ medium and small plate: `exp_137` @ 52K (i.e. `policy_checkpoint_0005200000`)
- ours w/o delay model `exp_147` @ 64K
- ours w/o history `exp_157` @ 64K
- ours w/o noise `exp_148` @ 64K
- ours w/ 16 steps history `exp_158` @ 64K
- recurrent `exp_177` @ policy_checkpoint_0002025000
- with velocity observations `exp_181` @ policy_checkpoint_0006400000
- PPO: policy from Ankur, TF action server `g_T_ee['position']['x'] + 0.07`
- downward with medium plate: `vicinity_height_offset = 0.1`, `O_T_K_goal.position.x += 0.05` in `move_to_goal_vicinity.py`
- Lee et al (16, 16): `exp_173` @ 64K
  - for medium plate, TF action server `g_T_ee['position']['z'] - 0.02`
  - for medium plate, TF action server `g_T_ee['position']['z'] - 0.04`
- `softgrasp_ros_controllers/config/softgrasp_ros_controllers.yaml` `success_thresh_z`:
  - large (main) plate `-0.02`
  - medium plate `-0.04`
  - small plate `-0.04`
- Also for the small plate experiment:
  - `run_experiment.launch`: `vicinity_height_offset = 0.07` and use `static_poses_far.launch`
  - translational stiffness = 2400, rotational space = 1700
  - TF action server `g_T_ee['position']['z'] - 0.04`
- ours cup
  - `g_T_ee['position']['x'] + 0.09`
  - `g_T_ee['position']['y'] - 0.05`
  - `g_T_ee['position']['z'] - 0.15`
  - `success_thresh_z` -0.08
  - gains: 2400, 1700
- lee et al (16, 16) cup
  - `g_T_ee['position']['x']`
  - `g_T_ee['position']['y'] + 0.01`
  - `g_T_ee['position']['z'] - 0.2`
  - `success_thresh_z` -0.08
  - gains: 2400, 1700
- random cup
  - move to goal vicinity `O_T_K_goal.position.x += 0.05`
  - force_limit 30
  - `g_T_ee['position']['x'] + 0.05`
  - `g_T_ee['position']['y']`
  - `g_T_ee['position']['z'] - 0.2`
