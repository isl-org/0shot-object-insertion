#!/usr/bin/env python
"""
Grasp plate and get ready for moving to goal
"""
import actionlib
from copy import deepcopy
import rospy
from softgrasp_ros.moveit_utils import ArmMover
import softgrasp_ros.transform_utils as tutils
from softgrasp_ros_controllers.msg import ManageExperimentAction, ManageExperimentGoal
import tf2_ros
from geometry_msgs.msg import TransformStamped
import sys


if __name__ == '__main__':
    rospy.init_node('grasp_plate', anonymous=True)
    try:
        ready_y_offset = rospy.get_param('~ready_y_offset')
        do_grasp = rospy.get_param('~do_grasp')
        joint_mode = rospy.get_param('~joint_mode')
    except KeyError as e:
        rospy.logerr(e)
        raise
    mover = ArmMover(display_trajectories=True)
    
    rospy.loginfo('Grasp Plate: Waiting for manage_experiment action...')
    manage_experiment = actionlib.SimpleActionClient('/manage_experiment', ManageExperimentAction)
    manage_experiment.wait_for_server()
    rospy.loginfo('Grasp Plate: Found manage_experiment action')

    joints_grasp = [
        -0.13022203851896416, 0.1719687740279917, -0.18047288030743192, -1.7422576071935632, -1.6365956127246222,
        1.950250840889082, 1.321040803644392
    ]
    joints_ready = [
        0.06270084013834927, 0.15246774997209248, 0.2498557462042201, -1.6635971532818892, -1.5123309957583748,
        1.3385234919256634, 1.225676951462108
    ]
    
    if do_grasp:
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)

        O_T_grasp_K = TransformStamped()
        rospy.loginfo('Waiting for panda_link0 -> grasp transform')
        O_T_grasp_K = tfBuffer.lookup_transform('panda_link0', 'grasp', rospy.Time.now(), rospy.Duration(10.0))
        rospy.loginfo('Found panda_link0 -> grasp transform')
        O_T_grasp_K = tutils.tfstamped_to_gpose(O_T_grasp_K)
        
        O_T_pregrasp_K = deepcopy(O_T_grasp_K)
        O_T_pregrasp_K.position.y += ready_y_offset
        O_T_ready_K = deepcopy(O_T_pregrasp_K)
        O_T_ready_K.position.z += 0.05
        
        O_T_pregrasp_EE = tutils.GPose(O_T_pregrasp_K) * tutils.get_K_T_EE()
        O_T_grasp_EE = tutils.GPose(O_T_grasp_K) * tutils.get_K_T_EE()
        O_T_ready_EE = tutils.GPose(O_T_ready_K) * tutils.get_K_T_EE()
        
        if not mover.eef_motion(O_T_pregrasp_EE.get_gpose(), relative=False, get_confirmation=True):
            rospy.logerr('Grasp Plate: failed motion 1')
            sys.exit(-1)
        done = mover.grasp_joint_mode(joints_grasp, get_confirmation=False) if joint_mode else mover.grasp(O_T_grasp_EE.get_gpose(), relative=False, get_confirmation=True)
        if not done:
            rospy.logerr('Grasp Plate: failed motion 2')
            sys.exit(-1)
        done = mover.eef_motion_joint_mode(joints_ready, get_confirmation=False) if joint_mode else mover.eef_motion(O_T_ready_EE.get_gpose(), relative=False, get_confirmation=True)
        if not done:
            rospy.logerr('Grasp Plate: failed motion 3')
            sys.exit(-1)
        if not mover.home(get_confirmation=False):
            rospy.logerr('Grasp Plate: failed motion 4')
            sys.exit(-1)

    goal = ManageExperimentGoal(stage_id=0)  # move to goal vicinity
    manage_experiment.send_goal(goal)
    rospy.loginfo("####### Called move to goal vicinity action action")
