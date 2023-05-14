import sys
import rospy
import moveit_commander
import transform_utils as tutils
from moveit_msgs.msg import DisplayTrajectory
from actionlib_msgs.msg import GoalStatusArray
from transform_utils import PoseBroadcaster
from softgrasp_ros.srv import SetGraspState, SetGraspStateRequest, SetGraspStateResponse
from std_msgs.msg import Empty


class ArmMover(object):
  def __init__(self, display_trajectories=True):
    self.display_trajectories = display_trajectories
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    moveit_commander.roscpp_initialize(sys.argv)
    self.robot = moveit_commander.RobotCommander()
    self.scene = moveit_commander.PlanningSceneInterface()
    self.move_group = moveit_commander.MoveGroupCommander('panda_arm')
    self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        DisplayTrajectory,
                                                        queue_size=20)
    self.planning_frame = self.move_group.get_planning_frame()
    self.eef_link = self.move_group.get_end_effector_link()
    self.pose_broadcaster = PoseBroadcaster()
    
    service_name = '/grasp_state'
    rospy.loginfo('Waiting for service %s...' % service_name)
    self.set_grasp_state = rospy.ServiceProxy(service_name, SetGraspState)
    rospy.loginfo('Found service %s' % service_name)


  def _plan_display_go(self, get_confirmation):
    success, plan, _, _ = self.move_group.plan()
    if not success:
      rospy.logerr("Planning failed")
      return False
    execute = True
    if self.display_trajectories or get_confirmation:
      execute = self._display_plan(plan, get_confirmation)
    if execute:
      rospy.loginfo('executing')
      self.move_group.execute(plan)
      self.move_group.stop()
      self.move_group.clear_pose_targets()
      # rospy.loginfo('Joints %s' % self.move_group.get_current_joint_values())
    else:
      rospy.loginfo('not executing')
    return execute


  def _display_plan(self, plan, get_confirmation):
    d = DisplayTrajectory()
    d.trajectory_start = self.robot.get_current_state()
    d.trajectory.append(plan)
    self.display_trajectory_publisher.publish(d)
    choice = 'y'
    if get_confirmation:
      choice = raw_input('Displaying planned trajectory. Execute? [yN]: ')
    return choice == 'y'


  def eef_motion(self, eef_pose, relative=False, get_confirmation=True):
    eef_pose = tutils.GPose(eef_pose)
    if relative:
      p = self.move_group.get_current_pose().pose
      p = tutils.GPose(p)
      eef_pose = p * eef_pose
    self.pose_broadcaster.broadcast(eef_pose, self.planning_frame, '{:s}_goal'.format(self.eef_link))
    self.move_group.set_pose_target(eef_pose.get_gpose())
    count = 0
    while count < 5:
      if self._plan_display_go(get_confirmation):
        break
      else:
        count += 1
    else:
      return False
    return True


  def eef_motion_joint_mode(self, joints, get_confirmation=True):
    self.move_group.set_joint_value_target(joints)
    count = 0
    while count < 5:
      if self._plan_display_go(get_confirmation):
        break
      else:
        count += 1
    else:
      return False
    return True


  def grasp(self, eef_pose, relative=False, get_confirmation=True):
    rospy.sleep(0.5)
    req = SetGraspStateRequest(state=False)
    if not self.set_grasp_state(req).success:
      rospy.logerr('Could not release grasp')
      return False
    if not self.eef_motion(eef_pose, relative=relative, get_confirmation=get_confirmation):
      return False
    choice = raw_input('do grasp? yN ') if get_confirmation else 'y'
    if choice == 'y':
      req = SetGraspStateRequest(state=True)
      if not self.set_grasp_state(req).success:
        rospy.logerr('Could not do grasp')
        return False
      else:
        return True
    else:
      return False
    # return self.home()

  
  def grasp_joint_mode(self, joints, get_confirmation=True):
    rospy.sleep(0.5)
    req = SetGraspStateRequest(state=False)
    if not self.set_grasp_state(req).success:
      rospy.logerr('Could not release grasp')
      return False
    if not self.eef_motion_joint_mode(joints, get_confirmation):
      return False
    choice = raw_input('do grasp? yN ') if get_confirmation else 'y'
    if choice == 'y':
      req = SetGraspStateRequest(state=True)
      if not self.set_grasp_state(req).success:
        rospy.logerr('Could not do grasp')
        return False
      else:
        return True
    else:
      return False
    # return self.home()
  
  
  def home(self, get_confirmation=True):
    self.move_group.set_named_target('ready')
    return self._plan_display_go(get_confirmation)