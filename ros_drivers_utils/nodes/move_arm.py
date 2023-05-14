#!/usr/bin/env python
"""
Move arm to end-effector pose if possible
"""
import rospy
from softgrasp_ros.moveit_utils import ArmMover
import numpy as np
import os
from tf import transformations as tx

osp = os.path


if __name__ == '__main__':
    rospy.init_node('arm_mover', anonymous=True)
    mover = ArmMover()
    filename = osp.join('~', 'research', 'softgrasp_perception', 'softgrasp_perception', 'data', 'grasp.txt')
    T = np.loadtxt(osp.expanduser(filename))
    # T = np.eye(4)
    # T = tx.euler_matrix(0, 0, np.deg2rad(-45))
    # T[0, 3] = 0.03
    mover.grasp(T, relative=False)  
    rospy.spin()
