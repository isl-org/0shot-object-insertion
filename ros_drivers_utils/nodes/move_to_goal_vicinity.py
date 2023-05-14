#!/usr/bin/env python
"""
Move arm to a pose above the "goal" frame
"""
import actionlib
import numpy as np
import rospy
from softgrasp_ros.moveit_utils import ArmMover
from softgrasp_ros.msg import MoveToGoalVicinityAction, MoveToGoalVicinityResult
import softgrasp_ros.transform_utils as tutils
from softgrasp_ros_controllers.msg import ManageExperimentAction, ManageExperimentGoal
import tf2_ros


eef_positions = np.array([
    [-0.278944, -0.154313, 0.525941],  # 2_4
    [-0.240433, -0.14585, 0.520486],   # 3_1
    [-0.204249, -0.137845, 0.511598],  # 4_1
])

joint_value_targets = [
    [-0.37129254787321825, -0.7894491013409015, 0.3699354033511982, -2.3543522969932282, 0.2885617387692133, 1.5736092789089446, 0.7626334942849083],
    [-0.34652809827787834, -0.7646193994927283, 0.2938729165938862, -2.36561901500397, 0.1896239932825168, 1.5929007437663774, 0.5641936437818398],
    [-0.6878974352677663, -0.8219327554870067, 0.44670270848225807, -2.3491078640452603, 0.2742256139340224, 1.5457363039641743, 0.3460168253120831],
]

class MoveToGoalVicinity:
    def __init__(self, height_offset, joint_mode):
        self.joint_mode = joint_mode
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self._as = actionlib.SimpleActionServer('/move_to_goal_vicinity', MoveToGoalVicinityAction, self.execute_cb,
                                                False)
        rospy.loginfo('MoveToGoalVicinity: Waiting for manage_experiment action...')
        self.manage_experiment = actionlib.SimpleActionClient('/manage_experiment', ManageExperimentAction)
        self.manage_experiment.wait_for_server()
        rospy.loginfo('MoveToGoalVicinity: Found manage_experiment action')

        # get vicinity pose
        rospy.loginfo('Waiting for panda_link0 -> goal transform')
        O_T_G = self.tfBuffer.lookup_transform('panda_link0', 'goal', rospy.Time.now(), rospy.Duration(10.0))
        C_T_G = self.tfBuffer.lookup_transform('camera_color_optical_frame', 'goal', rospy.Time.now(), rospy.Duration(10.0))
        rospy.loginfo('Found panda_link0 -> goal transform')
        O_T_K_goal = tutils.tfstamped_to_gpose(O_T_G)
        O_T_K_goal.position.z += height_offset
        self.O_T_EE_goal = tutils.GPose(O_T_K_goal) * tutils.get_K_T_EE()

        # look up the joint values
        query_eef_position = np.array([
            C_T_G.transform.translation.x,
            C_T_G.transform.translation.y,
            C_T_G.transform.translation.z,
        ])
        distances = np.linalg.norm(eef_positions - query_eef_position, axis=1)
        # print("###### distances", distances)
        self.joint_values_target = joint_value_targets[np.argmin(distances)]

        self._as.start()

    
    def execute_cb(self, goal):
        rospy.sleep(0.5)
        mover = ArmMover(display_trajectories=True)
        result = MoveToGoalVicinityResult()
        result.ok = mover.eef_motion_joint_mode(self.joint_values_target, get_confirmation=False) if self.joint_mode else \
            mover.eef_motion(self.O_T_EE_goal.get_gpose(), relative=False, get_confirmation=True)
        self._as.set_succeeded(result)
        if result.ok:
            rospy.loginfo("####### Moved to goal vicinity")
            goal = ManageExperimentGoal(stage_id=1)  # execute policy
            self.manage_experiment.send_goal(goal)
            rospy.loginfo("####### Called policy execution action")


if __name__ == '__main__':
    rospy.init_node('move_to_goal_vicinity', anonymous=True)
    try:
        height_offset = rospy.get_param('~height_offset')
        joint_mode = rospy.get_param('~joint_mode')
    except KeyError as e:
        rospy.logerr(e)
        raise

    mgv = MoveToGoalVicinity(height_offset, joint_mode)
    rospy.spin()
