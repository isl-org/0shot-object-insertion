#!/usr/bin/env python
import actionlib
from actionlib_msgs.msg import GoalStatusArray
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, SwitchControllerResponse
from softgrasp_ros.msg import MoveToGoalVicinityAction, MoveToGoalVicinityGoal, RaiseArmAction, RaiseArmGoal
from softgrasp_ros.srv import SetGraspState, SetGraspStateRequest, SetGraspStateResponse
from softgrasp_ros_controllers.msg import ManageExperimentAction, ManageExperimentResult
import rospy


class ExperimentManager:
  def __init__(self):
    rospy.loginfo('Starting release grasp action server')
    self.manage_experiment_server = actionlib.SimpleActionServer('/manage_experiment', ManageExperimentAction,
                                                                 self.manage_experiment_cb, False)
    
    service_name = '/controller_manager/switch_controller'
    rospy.loginfo('Waiting for service %s...' % service_name)
    self.switch_controllers = rospy.ServiceProxy(service_name, SwitchController)
    rospy.loginfo('Found service %s' % service_name)
    
    service_name = '/grasp_state'
    rospy.loginfo('Waiting for service %s...' % service_name)
    self.set_grasp_state = rospy.ServiceProxy(service_name, SetGraspState)
    rospy.loginfo('Found service %s' % service_name)

    action_name = '/raise_arm'
    rospy.loginfo('Waiting for action %s...' % action_name)
    self.raise_arm = actionlib.SimpleActionClient(action_name, RaiseArmAction)
    self.raise_arm.wait_for_server()
    rospy.loginfo('Found action %s' % service_name)
    
    self.manage_experiment_server.start()
    rospy.loginfo('Started release grasp action server')
    
    action_name = '/move_to_goal_vicinity'
    rospy.loginfo('Waiting for action %s...' % action_name)
    self.move_to_goal_vicinity = actionlib.SimpleActionClient(action_name, MoveToGoalVicinityAction)
    self.move_to_goal_vicinity.wait_for_server()
    rospy.loginfo('Found action %s' % service_name)
    
    # self.sub = rospy.Subscriber('execute_trajectory/status', GoalStatusArray, self.vicinity_cb)


  def manage_experiment_cb(self, goal):
    result = ManageExperimentResult()
    rospy.loginfo('##### Experiment management with %s' % goal)
    
    if goal.stage_id == 0:
      rospy.loginfo('Moving to goal vicinity')
      self.move_to_goal_vicinity.send_goal(MoveToGoalVicinityGoal())
      if not self.move_to_goal_vicinity.wait_for_result(rospy.Duration(60.0)):
        result.ok = False
        rospy.logerr('Could not move to goal vicinity')
        self.manage_experiment_server.set_succeeded(result)
        return
    elif goal.stage_id == 1:
      rospy.loginfo('Starting policy execution')
      req = SwitchControllerRequest(
        start_controllers=['tf_policy_controller', ],
        stop_controllers=['cartesian_impedance_example_controller', ],
        strictness=0,
        start_asap=False,
        timeout=0.0
      )
      if not self.switch_controllers(req).ok:
        result.ok = False
        rospy.logerr('Could not switch controller')
        self.manage_experiment_server.set_succeeded(result)
        return
    elif goal.stage_id == 2:
      rospy.sleep(2.0)
      rospy.loginfo('Releasing grasp')
      req = SetGraspStateRequest(state=False)
      if not self.set_grasp_state(req).success:
        result.ok = False
        rospy.logerr('Could not release grasp')
        self.manage_experiment_server.set_succeeded(result)
        return
      rospy.loginfo('Stopping controller')
      req = SwitchControllerRequest(
        start_controllers=['cartesian_impedance_example_controller'],
        stop_controllers=['tf_policy_controller', ],
        strictness=0,
        start_asap=False,
        timeout=0.0
      )
      if not self.switch_controllers(req).ok:
        result.ok = False
        rospy.logerr('Could not stop policy controller')
        self.manage_experiment_server.set_succeeded(result)
        return
      rospy.loginfo('Raising arm')
      self.raise_arm.send_goal(RaiseArmGoal())
      if not self.raise_arm.wait_for_result(rospy.Duration(60.0)):
        result.ok = False
        rospy.logerr('Could not raise arm')
        self.manage_experiment_server.set_succeeded(result)
        return
    else:
      result.ok = False
      rospy.logerr('Unknown stage_id %d' % goal.stage_id)
      self.manage_experiment_server.set_succeeded(result)
      return
    
    result.ok = True
    self.manage_experiment_server.set_succeeded(result)

  
  # def vicinity_cb(self, msg):
  #   if msg.status_list:
  #     if msg.status_list[0].status == 3:
  #       rospy.loginfo('EEF at goal vicinity')
  #       req = SwitchControllerRequest(
  #         start_controllers=['tf_policy_controller', ],
  #         stop_controllers=['cartesian_impedance_example_controller', ],
  #         strictness=0,
  #         start_asap=False,
  #         timeout=0.0
  #       )
  #       if self.switch_controllers(req).ok:
  #         self.controllers_switched = True
  #         self.sub.unregister()
  #       else:
  #         rospy.logerr('Could not switch controller')
        


if __name__ == '__main__':
  rospy.init_node('experiment_manager')
  em = ExperimentManager()
  rospy.spin()