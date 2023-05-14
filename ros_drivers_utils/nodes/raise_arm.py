#!/usr/bin/env python
"""
Raise arm
"""
import actionlib
import rospy
from softgrasp_ros.moveit_utils import ArmMover
from softgrasp_ros.msg import RaiseArmAction, RaiseArmResult
import softgrasp_ros.transform_utils as tutils
import tf2_ros
from geometry_msgs.msg import TransformStamped



class RaiseArm:
    def __init__(self, raise_amount):
        self.raise_amount = raise_amount
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self._as = actionlib.SimpleActionServer('/raise_arm', RaiseArmAction, self.execute_cb, False)
        self._as.start()

    
    def execute_cb(self, goal):
        # get current arm pose
        O_T_K = TransformStamped()
        rospy.loginfo('Waiting for panda_link0 -> panda_K transform')
        O_T_K = self.tfBuffer.lookup_transform('panda_link0', 'panda_K', rospy.Time.now(), rospy.Duration(10.0))
        rospy.loginfo('Found panda_link0 -> panda_K transform')
        
        O_T_K_goal = tutils.tfstamped_to_gpose(O_T_K)
        O_T_K_goal.position.z += self.raise_amount
        self.O_T_EE_goal = tutils.GPose(O_T_K_goal) * tutils.get_K_T_EE()
        
        result = RaiseArmResult()
        mover = ArmMover(display_trajectories=True)
        # result.ok = mover.eef_motion(self.O_T_EE_goal.get_gpose(), relative=False, get_confirmation=False)
        result.ok = mover.home(get_confirmation=False)
        self._as.set_succeeded(result)
        rospy.loginfo('Raised arm')


if __name__ == '__main__':
    rospy.init_node('move_to_goal_vicinity', anonymous=True)
    try:
        height_offset = rospy.get_param('~raise_amount')
    except KeyError as e:
        rospy.logerr(e)
        raise

    ra = RaiseArm(height_offset)
    rospy.spin()
